import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler
from xfuser.compact.compress_lowrank import subspace_iter

def quantize_1bit(
    input_tensor: torch.Tensor,
    rank
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor signs to 1-bit and calculate rank-k scale approximation.
    Packs 8 sign bits into one uint8.
    Calculates the U (N, K) and V (K, C) factors for scale approximation.
    
    Args:
        input_tensor: Input tensor (FP16), shape (n_tokens, channel)
        rank: Rank for scale approximation (default: 1)
        
    Returns:
        Tuple of (packed_tensor, scale_u, scale_v):
            - packed_tensor: Packed tensor (UINT8) containing binary values (C, N//8)
            - scale_u: Rank-k factor U (FP16), shape (N, K)
            - scale_v: Rank-k factor V (FP16), shape (K, C)
    """
    assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    N, C = input_tensor.shape
    assert C % 8 == 0, "Channel dimension C must be divisible by 8 for packing"
    assert rank >= 1

    # Calculate rank-k approximation of abs(input)
    # subspace_iter expects (C, N), returns U_ck(C, K), V_t_kn(K, N)
    with Profiler.scope(f"compact.quant.scale_rank{rank}_approx"):
        scale_V_ck, scale_U_t_kn, _ = subspace_iter(
            torch.abs(input_tensor).transpose(0, 1).contiguous(),
            rank=rank,
            num_iters=2
        )
    # Get scale_u (N, K) and scale_v (K, C)
    scale_u_nk = scale_U_t_kn.transpose(0, 1).contiguous().to(torch.half)
    scale_v_kc = scale_V_ck.transpose(0, 1).contiguous().to(torch.half)
    assert scale_u_nk.shape == (N, rank)
    assert scale_v_kc.shape == (rank, C)

    # --- Pack Signs using Kernel ---
    # Transpose input to (C, N) for kernel
    input_transposed = input_tensor.transpose(0, 1).contiguous()
    
    # Allocate output for packed bits
    packed_output = torch.empty((C, N // 8), dtype=torch.uint8, device=input_tensor.device).contiguous()
    
    # Kernel grid dimensions (based on transposed input C, N)
    BLOCK_SIZE_N_PACKED = 256 # Block size for the packed N dimension
    grid = (C, triton.cdiv(N // 8, BLOCK_SIZE_N_PACKED))

    with Profiler.scope("compact.quantize_1bit_kernel"):
        # Call kernel to pack the signs
        _quantize_1bit_kernel[grid](
            input_ptr=input_transposed,
            output_ptr=packed_output,
            A=C,  # Number of channels is grid dim 0
            B_8=N, # Number of tokens is grid dim 1 (unpacked)
            # Pass the block size used in grid calculation
            BLOCK_SIZE_B_PACKED=BLOCK_SIZE_N_PACKED, 
        )
        
    return packed_output, scale_u_nk, scale_v_kc

@triton.jit
def _quantize_1bit_kernel(
    input_ptr, # Input (C, N) FP16
    output_ptr, # Output (C, N//8) UINT8
    A, # = C
    B_8, # = N
    BLOCK_SIZE_B_PACKED: tl.constexpr = 256, # Block size for N//8 dim
):
    """
    Packs signs of input tensor (1 if >=0 else 0) into uint8.
    Input layout: (C, N), Output layout: (C, N//8)
    Grid: (C, cdiv(N//8, BLOCK_SIZE_B_PACKED))
    """
    # Kernel logic remains the same, just packs signs.
    pid_a = tl.program_id(0) # Channel C
    pid_b_packed = tl.program_id(1) # Block index for N//8 dimension

    B_packed = B_8 // 8 # N//8

    b_packed_offset = pid_b_packed * BLOCK_SIZE_B_PACKED
    offs_b_packed = b_packed_offset + tl.arange(0, BLOCK_SIZE_B_PACKED)
    mask_b_packed = offs_b_packed < B_packed

    b_8_offset = b_packed_offset * 8
    packed_result = tl.zeros((BLOCK_SIZE_B_PACKED,), dtype=tl.uint8)

    for i in range(8):
        offs_b_8_for_bit_i = b_8_offset + tl.arange(0, BLOCK_SIZE_B_PACKED) * 8 + i
        input_offsets = pid_a * B_8 + offs_b_8_for_bit_i
        # Mask ensures we only consider elements within the valid packed block range *and* within N
        load_mask = mask_b_packed & (offs_b_8_for_bit_i < B_8)
        
        x = tl.load(input_ptr + input_offsets, mask=load_mask, other=0.0)
        # Determine the binary value (1 if x >= 0, else 0)
        binary_value = tl.where(x >= 0, 1, 0).to(tl.uint8) & tl.where(load_mask, 1, 0).to(tl.uint8)
        
        shifted_binary = (binary_value << i).to(tl.uint8)
        packed_result = (packed_result | shifted_binary).to(tl.uint8)

    output_ptr_base = output_ptr + pid_a * B_packed
    output_offsets = output_ptr_base + offs_b_packed
    tl.store(output_offsets, packed_result, mask=mask_b_packed)

def dequantize_1bit(
    packed_tensor: torch.Tensor, # Packed bits (C, N//8) UINT8
    scale_u: torch.Tensor,       # Scale factor U (N, K) FP16
    scale_v: torch.Tensor        # Scale factor V (K, C) FP16
) -> torch.Tensor:
    """
    Dequantize packed 1-bit sign values back to FP16 using rank-k scale factors.
    Computes the full scale matrix S = U @ V internally and passes it to the kernel.
    Uses Triton kernel for unpacking and scaling.
    
    Args:
        packed_tensor: Packed tensor (UINT8) shape (C, N//8)
        scale_u: Rank-k factor U (FP16), shape (N, K)
        scale_v: Rank-k factor V (FP16), shape (K, C)
        
    Returns:
        Dequantized tensor (FP16) shape (N, C)
    """
    assert packed_tensor.dtype == torch.uint8, "Packed tensor must be UINT8"
    assert scale_u.dtype == torch.half, "Scale U must be FP16"
    assert scale_v.dtype == torch.half, "Scale V must be FP16"
    assert scale_u.ndim == 2, "Scale U must be 2D (N, K)"
    assert scale_v.ndim == 2, "Scale V must be 2D (K, C)"
    assert packed_tensor.ndim == 2, "Packed tensor must be 2D"
    assert scale_u.shape[1] == scale_v.shape[0], f"Rank K mismatch: U K dim {scale_u.shape[1]} != V K dim {scale_v.shape[0]}"

    packed_tensor = packed_tensor.contiguous()
    scale_u = scale_u.contiguous() # (N, K)
    scale_v = scale_v.contiguous() # (K, C)

    C, N_8 = packed_tensor.shape # C = channels, N_8 = packed tokens
    N = N_8 * 8 # Unpacked tokens
    rank = scale_u.shape[1] # Infer rank K

    assert scale_u.shape[0] == N, f"Scale U N dim {scale_u.shape[0]} must match unpacked N dim {N}"
    assert scale_v.shape[1] == C, f"Scale V C dim {scale_v.shape[1]} must match packed C dim {C}"

    # --- Compute Full Scale Matrix (PyTorch) ---
    # S = U @ V -> (N, K) @ (K, C) -> (N, C)
    with Profiler.scope(f"compact.dequant.scale_rank{rank}_matmul"):
        scale_matrix_nc = (scale_u @ scale_v).to(torch.half)
    assert scale_matrix_nc.shape == (N, C)

    # --- Prepare Scale for Kernel ---
    # Kernel expects scale matrix in (C, N) layout
    scale_matrix_cn = scale_matrix_nc.transpose(0, 1).contiguous()

    # Allocate output in (C, N) layout for kernel
    output_cn = torch.empty((C, N), dtype=torch.half, device=packed_tensor.device)
    
    # Define the block size for the packed dimension (N//8)
    BLOCK_SIZE_N_PACKED = 256 
    # Grid dimensions: (C, cdiv(N//8, BLOCK_SIZE_N_PACKED))
    grid = (C, triton.cdiv(N_8, BLOCK_SIZE_N_PACKED))

    with Profiler.scope("compact.dequantize_1bit_kernel"):
        _dequantize_1bit_kernel[grid](
            input_ptr=packed_tensor, # (C, N//8)
            output_ptr=output_cn,    # (C, N)
            scale_matrix_ptr=scale_matrix_cn, # Pass pre-computed scale (C, N)
            A=C,                     # Num Channels
            B=N_8,                   # Num Packed Elements per channel
            B_8=N,                   # Num Unpacked Elements per channel
            stride_scale_c=scale_matrix_cn.stride(0),
            stride_scale_n=scale_matrix_cn.stride(1),
            BLOCK_SIZE_B=BLOCK_SIZE_N_PACKED, # Block size for packed dimension N//8
        )
        
    # Transpose output from (C, N) back to (N, C)
    output_nc = torch.transpose(output_cn, 0, 1).contiguous()
    return output_nc

@triton.jit
def _dequantize_1bit_kernel(
    input_ptr,      # Packed bits (C, N//8) UINT8
    output_ptr,     # Output (C, N) FP16
    scale_matrix_ptr, # <<< Pre-computed scale (C, N) FP16
    A,  # = C (Number of channels)
    B,  # = N//8 (Number of packed elements per channel)
    B_8,# = N (Number of unpacked elements per channel)
    stride_scale_c, # <<< Stride for scale C dim
    stride_scale_n, # <<< Stride for scale N dim
    BLOCK_SIZE_B: tl.constexpr = 256, # Block size for the packed dimension N//8
):
    """
    Dequantizes packed 1-bit signs to FP16 using pre-computed scale matrix S.
    Input layout: packed (C, N//8), scale (C, N)
    Output layout: (C, N)
    Grid: (C, cdiv(N//8, BLOCK_SIZE_B))
    """
    # Program IDs
    pid_a = tl.program_id(0)  # Channel ID (C)
    pid_b = tl.program_id(1)  # Block ID in the packed dimension N//8

    # Calculate offsets for the current block in the packed dimension
    b_offset = pid_b * BLOCK_SIZE_B
    offs_b = b_offset + tl.arange(0, BLOCK_SIZE_B)
    mask_b = offs_b < B  # Mask for packed dimension elements

    # Calculate pointer offsets for loading packed data
    packed_ptrs = input_ptr + pid_a * B + offs_b
    # Load packed data for the current block
    packed_data = tl.load(packed_ptrs, mask=mask_b, other=0) # Shape: [BLOCK_SIZE_B]

    # Calculate base offset for output pointer (start of the row in C, N layout)
    output_row_start_ptr = output_ptr + pid_a * B_8 # Offset to the start of channel pid_a
    # Calculate base offset for scale matrix pointer
    scale_row_start_ptr = scale_matrix_ptr + pid_a * stride_scale_c

    # Unpack 8 bits and dequantize
    for i in range(8):
        # --- Unpack Bit ---
        bits = ((packed_data >> i) & 1).to(tl.int8) # Shape: [BLOCK_SIZE_B]
        signs = tl.where(bits == 1, 1.0, -1.0).to(tl.float16) # Convert bits to +1/-1 signs

        # --- Load Scale from Matrix ---
        # Calculate corresponding unpacked N offsets for this bit position
        offs_b_8 = offs_b * 8 + i # Offset within the unpacked row (N dimension)
        # Create mask for the unpacked dimension (relative to the block start)
        mask_b_8 = mask_b & (offs_b_8 < B_8) # Ensure we are within C, N bounds
        
        # Load scale values from the pre-computed matrix (C, N)
        scale_ptrs = scale_row_start_ptr + offs_b_8 * stride_scale_n
        scale_block = tl.load(scale_ptrs, mask=mask_b_8, other=0.0).to(tl.float16)

        # --- Dequantize ---
        scaled = signs * scale_block

        # --- Store Output ---
        # Calculate output pointers for the current bit position within the unpacked row
        output_ptrs = output_row_start_ptr + offs_b_8
        # Store the dequantized values, masking invalid elements using unpacked mask
        tl.store(output_ptrs, scaled, mask=mask_b_8)

def sim_binary(input_tensor: torch.Tensor, rank: int = 1) -> torch.Tensor:
    """
    Simulates channel-wise 1-bit quantization using rank-k scale approximation.
    Args:
        input_tensor: The input tensor (N, C).
        rank: The rank for scale approximation.
    Returns:
        A tensor of the same size as the input, representing the dequantized result.
    """
    # NOTE: must use mean, otherwise the dequantized tensor's norm is too large, resulting nonsensical output
    # scale = torch.mean(torch.abs(input_tensor), dim=tuple(range(input_tensor.ndim - 1)), keepdim=True) if scale is None else scale
    from xfuser.compact.compress_lowrank import svd, subspace_iter
    # # u, v = svd(torch.abs(input_tensor), 2)
    # We input with a transposed shape (C, N) to align with fastpath
    # But this shape change does not affect u and v as we swap them back
    RANK=rank
    v, u, _ = subspace_iter(torch.abs(input_tensor).transpose(0, 1), RANK, 2)
    u = u.transpose(0, 1)
    v = v.transpose(0, 1)
    scale = u @ v
    assert scale.dtype == torch.half, "Scale must be FP16"
    # Quantize to -1 or 1 based on the sign
    quantized_tensor = torch.sign(input_tensor)
    # Handle zeros by converting them to 1
    # This ensures that zeros in the input are represented as 1 in the quantized tensor
    quantized_tensor = torch.where(quantized_tensor == 0, torch.ones_like(quantized_tensor), quantized_tensor)
    # Dequantize by multiplying with the scale
    dequantized_tensor = quantized_tensor * scale
    return dequantized_tensor

@torch.compile
def sim_ternary(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Simulates channel-wise ternary (-1, 0, 1) quantization with per-channel scale.
    This is a simulation function for a 3-value compression type.

    Args:
        input_tensor: The input tensor. The last dimension is assumed to be the channel dimension.

    Returns:
        A tensor of the same size as the input, representing the dequantized result.
    """
    # Calculate scale using mean absolute value, same as binary quantization
    scale = torch.mean(torch.abs(input_tensor), dim=tuple(range(input_tensor.ndim - 1)), keepdim=True)
    assert scale.dtype == torch.half, "Scale must be FP16"
    
    # Use threshold of 0.5*scale to determine if value should be 0 or ±1
    threshold = 0.5 * scale
    
    # Create a zero tensor with the same shape
    zero_tensor = torch.zeros_like(input_tensor)
    
    # Quantize to -1, 0, or 1 based on thresholds
    quantized_tensor = torch.where(input_tensor > threshold, torch.ones_like(input_tensor), zero_tensor)
    quantized_tensor = torch.where(input_tensor < -threshold, -torch.ones_like(input_tensor), quantized_tensor)
    
    # Dequantize by multiplying with the scale
    dequantized_tensor = quantized_tensor * scale
    
    return dequantized_tensor

@Profiler.prof_func("compact.quantize_int2")
@torch.jit.script
def quantize_int2(
    input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a 2D FP16 tensor to INT2 using mean scaling, packing 4 values into a UINT8.
    Maps values based on channel-wise mean absolute value.
    
    Args:
        input_tensor: 2D tensor (N, C) with dtype torch.half. C must be divisible by 4.
        
    Returns:
        Tuple of (packed_uint8, scale):
            - packed_uint8: Packed tensor (N, C // 4) with dtype torch.uint8
            - scale: Channel-wise scale factor (1, C) with dtype torch.half
    """
    if input_tensor.dim() != 2:
        raise ValueError(f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions.")
    if input_tensor.dtype != torch.half:
        raise ValueError(f"Input tensor must be torch.half, but got {input_tensor.dtype}.")

    N, C = input_tensor.shape
    if C % 4 != 0:
        raise ValueError(f"Channel dimension C ({C}) must be divisible by 4 for INT2 packing.")

    input_float32 = input_tensor.float()
    scale = torch.mean(torch.abs(input_float32), dim=0, keepdim=True)
    assert scale.size(1) == C
    neg_scale = -scale

    cond_1 = (input_float32 >= neg_scale) & (input_float32 < 0)
    cond_2 = (input_float32 >= 0) & (input_float32 <= scale)
    cond_3 = input_float32 > scale

    int2_vals = (cond_1.to(torch.uint8) * 1 +
                 cond_2.to(torch.uint8) * 2 +
                 cond_3.to(torch.uint8) * 3)

    reshaped_int2 = int2_vals.view(N, C // 4, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=input_tensor.device).view(1, 1, 4)
    shifted_values = reshaped_int2 << shifts
    packed_uint8 = torch.sum(shifted_values, dim=-1, dtype=torch.uint8)
    packed_uint8 = packed_uint8.view(N, C // 4)
    return packed_uint8, scale.half()

@Profiler.prof_func("compact.dequantize_int2")
@torch.jit.script
def dequantize_int2(
    packed_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Unpacks UINT8 tensor containing 4xINT2 values and dequantizes to FP16.
    
    Args:
        packed_tensor: Packed tensor (N, C // 4) with dtype torch.uint8
        scale: Channel-wise scale factor (C) with dtype torch.half
        
    Returns:
        Dequantized tensor (N, C) with dtype torch.half
    """
    assert packed_tensor.dtype == torch.uint8, f"Packed tensor must be torch.uint8, but got {packed_tensor.dtype}."
    assert packed_tensor.ndim == 2, f"Packed tensor must be 2D, but got {packed_tensor.ndim} dims."
    assert scale.dtype == torch.half, f"Scale tensor must be torch.half, but got {scale.dtype}."
    assert scale.ndim == 1, f"Scale shape must be (C), but got {scale.shape}"
    scale = scale.view(1, -1)
    N, packed_C = packed_tensor.shape
    # original_c: Original channel dimension (must be divisible by 4)
    original_c = packed_C * 4
    assert scale.shape[1] == original_c, f"Scale({scale.shape}) channel dim C ({scale.shape[1]}) must match packed({packed_tensor.shape}) original_c ({original_c})"

    packed_expanded = packed_tensor.unsqueeze(-1)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed_tensor.device).view(1, 1, 4)
    mask = 3
    unpacked_int2 = ((packed_expanded >> shifts) & mask).to(torch.uint8)
    unpacked_reshaped = unpacked_int2.view(N, original_c)

    level_pp = 2.0 * scale
    level_p  = 0.5 * scale
    level_n  = -0.5 * scale
    level_nn = -2.0 * scale

    dequantized_tensor = torch.zeros((N, original_c), dtype=torch.half, device=packed_tensor.device)

    dequantized_tensor = torch.where(unpacked_reshaped == 0, level_nn, dequantized_tensor)
    dequantized_tensor = torch.where(unpacked_reshaped == 1, level_n, dequantized_tensor)
    dequantized_tensor = torch.where(unpacked_reshaped == 2, level_p, dequantized_tensor)
    dequantized_tensor = torch.where(unpacked_reshaped == 3, level_pp, dequantized_tensor)

    return dequantized_tensor

@torch.compile
def sim_int2(input_tensor: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
    """
    Simulates channel-wise INT2 quantization with mean-based scaling.
    This is a simulation function for the INT2 compression type.
    
    Maps input values to 4 levels based on per-channel mean absolute value:
    [-2*mean_abs, -0.5*mean_abs, +0.5*mean_abs, +2*mean_abs]
    
    Args:
        input_tensor: 2D tensor (N, C) with dtype torch.half
        
    Returns:
        Simulated dequantized tensor with same shape and dtype
    """
    assert input_tensor.dim() == 2, f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions."
    assert input_tensor.dtype == torch.half, f"Input tensor must be torch.half, but got {input_tensor.dtype}."
    input_dtype = input_tensor.dtype

    reduce_dims = 0
    input_float32 = input_tensor.float()

    scale = torch.mean(torch.abs(input_float32), dim=reduce_dims, keepdim=True) if scale is None else scale

    level_pp = 2.0 * scale
    level_p  = 0.5 * scale
    level_n  = -0.5 * scale
    level_nn = -2.0 * scale

    output = torch.zeros_like(input_float32)

    output = torch.where((input_float32 >= 0) & (input_float32 <= scale), level_p, output)
    output = torch.where(input_float32 > scale, level_pp, output)
    output = torch.where((input_float32 < 0) & (input_float32 >= -scale), level_n, output)
    output = torch.where(input_float32 < -scale, level_nn, output)

    dequantized_tensor = output.to(input_dtype)
    return dequantized_tensor

@torch.jit.script
def quantize_int8(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs channel-wise INT8 affine quantization on a 2D input tensor.

    Args:
        input_tensor: A 2D PyTorch tensor (e.g., shape [N, C]) of dtype float32.
                      The last dimension (-1) is treated as the channel dimension.

    Returns:
        A tuple containing:
        - quantized_tensor: The quantized tensor of dtype torch.int8.
        - scale: The scale factor tensor (one per channel), dtype float32.
        - zero_point: The zero point tensor (one per channel), dtype torch.int16.
    """
    assert input_tensor.dim() == 2, f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions."
    assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    # Define quantization range for INT8
    qmin = -128
    qmax = 127
    # Channel dimension is the last one (-1, which is 1 for 2D)
    channel_dim = -1
    # Reduce along non-channel dimensions (dim 0 for 2D)
    reduce_dim = 0 # Or tuple(range(input_tensor.dim() - 1)) for >2D
    min_val = torch.min(input_tensor, dim=reduce_dim, keepdim=True).values
    max_val = torch.max(input_tensor, dim=reduce_dim, keepdim=True).values
    # Add epsilon to prevent division by zero if min_val == max_val
    scale = (max_val - min_val) / (qmax - qmin + 1e-6)
    scale = scale.to(torch.half)
    # Calculate zero point (per channel)
    # zero_point = qmin - round(min_val / scale)
    # Ensure scale isn't exactly zero before dividing
    zero_point = qmin - torch.round(min_val / scale)
    # Clamp zero_point to the int8 range
    # Use a wider int type temporarily if needed, but int16 should suffice here.
    zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int16)
    # Perform quantization: q = round(r / scale + zero_point)
    quantized_data = torch.round(input_tensor / scale + zero_point)
    # Clamp the quantized values to the int8 range
    quantized_tensor = torch.clamp(quantized_data, qmin, qmax).to(torch.int8)
    assert quantized_tensor.dtype == torch.int8
    assert scale.dtype == torch.half
    assert zero_point.dtype == torch.int16
    return quantized_tensor, scale, zero_point

@torch.jit.script
def dequantize_int8(q_tensor, scale, zero_point):
     # Ensure scale/zp are broadcastable if they were squeezed
     if q_tensor.dim() > scale.dim():
         scale = scale.unsqueeze(0) # Add back the N dim for broadcasting
     if q_tensor.dim() > zero_point.dim():
        zero_point = zero_point.unsqueeze(0) # Add back the N dim
     # Formula: r = (q - zero_point) * scale
     # Cast quantized tensor to float for calculation
     dequantized_tensor = (q_tensor.half() - zero_point.half()) * scale
     assert dequantized_tensor.dtype == torch.half
     return dequantized_tensor