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
    Packs 8 sign bits from the channel dimension (C) into one uint8.
    Input tensor layout: (N, C)
    Packed tensor layout: (N, C//8)
    Calculates the U (N, K or N, 1) and V (K, C or 1, C) factors for scale approximation.
    
    Args:
        input_tensor: Input tensor (FP16), shape (N, C)
        rank: Rank for scale approximation (-1 for mean, >=1 for subspace).
        
    Returns:
        Tuple of (packed_tensor, scale_u, scale_v):
            - packed_tensor: Packed tensor (UINT8) containing binary values (N, C//8)
            - scale_u: Rank-k factor U (FP16), shape (N, K or N, 1 if rank=-1)
            - scale_v: Rank-k factor V (FP16), shape (K, C or 1, C if rank=-1)
    """
    assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    N, C = input_tensor.shape
    assert C % 8 == 0, "Channel dimension C must be divisible by 8 for packing"
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"

    # --- Scale Calculation (Input N, C) --- 
    input_abs_nc = torch.abs(input_tensor).contiguous()

    if rank == -1:
        # Use channel-wise mean as scale
        with Profiler.scope("compact.quant.scale_channel_mean"):
            # Calculate mean across N dimension for each channel C -> shape (C,)
            mean_scale_c = torch.mean(input_abs_nc, dim=0)
        # Reshape for U/V structure: u_nk (N, 1), v_kc (1, C)
        # scale_v (1, C) contains the mean scales
        scale_v_kc = mean_scale_c.unsqueeze(0).contiguous().to(torch.half) # (1, C)
        # scale_u (N, 1) contains ones
        # scale_u_nk = torch.ones((N, 1), device=input_tensor.device, dtype=torch.half) # (N, 1)
        scale_u_nk = torch.mean(input_abs_nc, dim=1, keepdim=True) # (N, 1)
        scale_u_nk = scale_u_nk / scale_u_nk.mean(dim=0, keepdim=True)
        effective_rank = 1 # For assertion checks later
    else: # rank >= 1
        # Calculate rank-k approximation of abs(input)
        # subspace_iter expects (Features=N, Samples=C), returns U_nk(N, K), V_t_kc(K, C)
        with Profiler.scope(f"compact.quant.scale_rank{rank}_approx"):
            # Pass (N, C) directly
            scale_U_nk, scale_V_t_kc, _ = subspace_iter(
                input_abs_nc,
                rank=rank,
                num_iters=2
            )
        # Get scale_u (N, K) and scale_v (K, C)
        scale_u_nk = scale_U_nk.contiguous().to(torch.half)       # Shape (N, K)
        scale_v_kc = scale_V_t_kc.contiguous().to(torch.half) # Shape (K, C)
        effective_rank = rank

    assert scale_u_nk.shape == (N, effective_rank)
    assert scale_v_kc.shape == (effective_rank, C)
  
    # --- Pack Signs using Kernel --- 
    # Input tensor is already (N, C)
    input_nc = input_tensor.contiguous()
    
    # Allocate output for packed bits (N, C//8)
    packed_output = torch.empty((N, C // 8), dtype=torch.uint8, device=input_tensor.device).contiguous()
    
    # Kernel grid dimensions (based on output N, C//8)
    BLOCK_SIZE_C_PACKED = 256 # Block size for the packed C dimension
    grid = (N, triton.cdiv(C // 8, BLOCK_SIZE_C_PACKED))

    with Profiler.scope("compact.quantize_1bit_kernel"):
        # Call kernel to pack the signs
        _quantize_1bit_kernel[grid](
            input_ptr=input_nc,
            output_ptr=packed_output,
            N_ROWS=N,  # Number of tokens is grid dim 0
            C_COLS_8=C, # Number of channels is grid dim 1 (unpacked)
            # Pass the block size used in grid calculation
            BLOCK_SIZE_C_PACKED=BLOCK_SIZE_C_PACKED, 
        )
        
    return packed_output, scale_u_nk, scale_v_kc

@triton.jit
def _quantize_1bit_kernel(
    input_ptr, # Input (N, C) FP16
    output_ptr, # Output (N, C//8) UINT8
    N_ROWS, # = N
    C_COLS_8, # = C
    BLOCK_SIZE_C_PACKED: tl.constexpr = 256, # Block size for C//8 dim
):
    """
    Packs signs of input tensor (1 if >=0 else 0) into uint8 along C dim.
    Input layout: (N, C), Output layout: (N, C//8)
    Grid: (N, cdiv(C//8, BLOCK_SIZE_C_PACKED))
    """
    # Kernel logic packs signs along C dimension.
    pid_n = tl.program_id(0) # Row N
    pid_c_packed = tl.program_id(1) # Block index for C//8 dimension

    C_COLS_PACKED = C_COLS_8 // 8 # C//8

    # Calculate offsets for the packed C dimension block
    c_packed_offset = pid_c_packed * BLOCK_SIZE_C_PACKED
    offs_c_packed = c_packed_offset + tl.arange(0, BLOCK_SIZE_C_PACKED)
    mask_c_packed = offs_c_packed < C_COLS_PACKED

    # Base offset for the unpacked C dimension
    c_8_offset = c_packed_offset * 8
    
    # Initialize packed result for the block
    packed_result = tl.zeros((BLOCK_SIZE_C_PACKED,), dtype=tl.uint8)

    # Iterate through the 8 bits to pack for each uint8 output element
    for i in range(8):
        # Calculate unpacked C offsets for the current bit position
        offs_c_8_for_bit_i = c_8_offset + tl.arange(0, BLOCK_SIZE_C_PACKED) * 8 + i
        
        # Calculate input pointer offsets: base = pid_n * C, add c offsets
        input_offsets = pid_n * C_COLS_8 + offs_c_8_for_bit_i
        
        # Create load mask: ensure c_packed is valid AND unpacked c is within C
        load_mask = mask_c_packed & (offs_c_8_for_bit_i < C_COLS_8)
        
        # Load input values
        x = tl.load(input_ptr + input_offsets, mask=load_mask, other=0.0)
        
        # Determine the binary value (1 if x >= 0, else 0)
        binary_value = tl.where(x >= 0, 1, 0).to(tl.uint8) 
        # Mask out contributions from invalid elements (using load_mask)
        binary_value = binary_value & tl.where(load_mask, 1, 0).to(tl.uint8)
        
        # Shift the binary value to its correct bit position
        shifted_binary = (binary_value << i).to(tl.uint8)
        
        # Combine (OR) the shifted value into the packed result
        packed_result = (packed_result | shifted_binary).to(tl.uint8)

    # Calculate output pointer base: base = pid_n * C_packed
    output_ptr_base = output_ptr + pid_n * C_COLS_PACKED
    # Calculate full output offsets
    output_offsets = output_ptr_base + offs_c_packed
    # Store the packed results for the block
    tl.store(output_offsets, packed_result, mask=mask_c_packed)

def dequantize_1bit(
    packed_tensor: torch.Tensor, # Packed bits (N, C//8) UINT8
    scale_u: torch.Tensor,       # Scale factor U (N, K) FP16
    scale_v: torch.Tensor        # Scale factor V (K, C) FP16
) -> torch.Tensor:
    """
    Dequantize packed 1-bit sign values (packed along C dim) back to FP16 using rank-k scale factors.
    Computes the full scale matrix S = U @ V internally and passes it to the kernel.
    Uses Triton kernel for unpacking and scaling.
    Input packed layout: (N, C//8)
    Output layout: (N, C)
    
    Args:
        packed_tensor: Packed tensor (UINT8) shape (N, C//8)
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
    assert packed_tensor.ndim == 2, "Packed tensor must be 2D (N, C//8)"
    assert scale_u.shape[1] == scale_v.shape[0], f"Rank K mismatch: U K dim {scale_u.shape[1]} != V K dim {scale_v.shape[0]}"

    packed_tensor = packed_tensor.contiguous() # (N, C//8)
    scale_u = scale_u.contiguous() # (N, K)
    scale_v = scale_v.contiguous() # (K, C)

    N, C_8 = packed_tensor.shape # N = tokens, C_8 = packed channels
    C = C_8 * 8 # Unpacked channels
    rank = scale_u.shape[1] # Infer rank K

    assert scale_u.shape[0] == N, f"Scale U N dim {scale_u.shape[0]} must match packed N dim {N}"
    assert scale_v.shape[1] == C, f"Scale V C dim {scale_v.shape[1]} must match unpacked C dim {C}"

    # --- Compute Full Scale Matrix (PyTorch) --- 
    # S = U @ V -> (N, K) @ (K, C) -> (N, C)
    with Profiler.scope(f"compact.dequant.scale_rank{rank}_matmul"):
        scale_matrix_nc = (scale_u @ scale_v).to(torch.half)
    assert scale_matrix_nc.shape == (N, C)

    # --- Prepare Scale for Kernel ---
    # Kernel expects scale matrix in (N, C) layout
    scale_matrix_nc = scale_matrix_nc.contiguous()

    # Allocate output in (N, C) layout for kernel
    output_nc = torch.empty((N, C), dtype=torch.half, device=packed_tensor.device)
    
    # Define the block size for the packed dimension (C//8)
    BLOCK_SIZE_C_PACKED = 256 
    # Grid dimensions: (N, cdiv(C//8, BLOCK_SIZE_C_PACKED))
    grid = (N, triton.cdiv(C_8, BLOCK_SIZE_C_PACKED))

    with Profiler.scope("compact.dequantize_1bit_kernel"):
        _dequantize_1bit_kernel[grid](
            input_ptr=packed_tensor, # (N, C//8)
            output_ptr=output_nc,    # (N, C)
            scale_matrix_ptr=scale_matrix_nc, # Pass pre-computed scale (N, C)
            N_ROWS=N,                 # Num Tokens
            C_COLS=C_8,               # Num Packed Channels per token
            C_COLS_8=C,               # Num Unpacked Channels per token
            stride_scale_n=scale_matrix_nc.stride(0), # Stride for scale N dim
            stride_scale_c=scale_matrix_nc.stride(1), # Stride for scale C dim
            BLOCK_SIZE_C=BLOCK_SIZE_C_PACKED, # Block size for packed dimension C//8
        )
        
    # Output is already (N, C)
    return output_nc

@triton.jit
def _dequantize_1bit_kernel(
    input_ptr,      # Packed bits (N, C//8) UINT8
    output_ptr,     # Output (N, C) FP16
    scale_matrix_ptr, # <<< Pre-computed scale (N, C) FP16
    N_ROWS,  # = N (Number of tokens)
    C_COLS,  # = C//8 (Number of packed elements per token)
    C_COLS_8,# = C (Number of unpacked elements per token)
    stride_scale_n, # <<< Stride for scale N dim
    stride_scale_c, # <<< Stride for scale C dim
    BLOCK_SIZE_C: tl.constexpr = 256, # Block size for the packed dimension C//8
):
    """
    Dequantizes packed 1-bit signs (packed along C) to FP16 using pre-computed scale matrix S.
    Input layout: packed (N, C//8), scale (N, C)
    Output layout: (N, C)
    Grid: (N, cdiv(C//8, BLOCK_SIZE_C))
    """
    # Program IDs
    pid_n = tl.program_id(0)  # Token ID (N)
    pid_c_packed = tl.program_id(1)  # Block ID in the packed dimension C//8

    # Calculate offsets for the current block in the packed dimension C//8
    c_packed_offset = pid_c_packed * BLOCK_SIZE_C
    offs_c_packed = c_packed_offset + tl.arange(0, BLOCK_SIZE_C)
    mask_c_packed = offs_c_packed < C_COLS  # Mask for packed dimension elements

    # Calculate pointer offsets for loading packed data for the current token (row pid_n)
    # Base ptr = input_ptr + pid_n * C_COLS (stride for packed dim)
    packed_row_start_ptr = input_ptr + pid_n * C_COLS
    packed_ptrs = packed_row_start_ptr + offs_c_packed
    # Load packed data for the current block
    packed_data = tl.load(packed_ptrs, mask=mask_c_packed, other=0) # Shape: [BLOCK_SIZE_C]

    # Calculate base offset for output pointer (start of the row in N, C layout)
    output_row_start_ptr = output_ptr + pid_n * C_COLS_8 # Offset to the start of token pid_n
    # Calculate base offset for scale matrix pointer (start of the row in N, C layout)
    scale_row_start_ptr = scale_matrix_ptr + pid_n * stride_scale_n

    # Base offset for the unpacked C dimension
    c_8_offset = c_packed_offset * 8

    # Unpack 8 bits and dequantize
    for i in range(8):
        # --- Unpack Bit --- 
        # Extract the i-th bit from each packed byte
        bits = ((packed_data >> i) & 1).to(tl.int8) # Shape: [BLOCK_SIZE_C]
        # Convert bits to +1/-1 signs
        signs = tl.where(bits == 1, 1.0, -1.0).to(tl.float16)

        # --- Load Scale from Matrix --- 
        # Calculate corresponding unpacked C offsets for this bit position
        offs_c_8 = c_8_offset + tl.arange(0, BLOCK_SIZE_C) * 8 + i # Offset within the unpacked row (C dimension)
        
        # Create mask for the unpacked dimension (relative to the block start)
        # Ensure we are within N, C bounds (mask_c_packed takes care of block bounds)
        mask_c_8 = mask_c_packed & (offs_c_8 < C_COLS_8) 
        
        # Load scale values from the pre-computed matrix (N, C)
        # Ptrs = base_row_ptr + c_offset * stride_c
        scale_ptrs = scale_row_start_ptr + offs_c_8 * stride_scale_c
        scale_block = tl.load(scale_ptrs, mask=mask_c_8, other=0.0).to(tl.float16)

        # --- Dequantize --- 
        scaled = signs * scale_block

        # --- Store Output --- 
        # Calculate output pointers for the current bit position within the unpacked row
        # Ptrs = base_row_ptr + c_offset
        output_ptrs = output_row_start_ptr + offs_c_8
        # Store the dequantized values, masking invalid elements using unpacked mask
        tl.store(output_ptrs, scaled, mask=mask_c_8)

def sim_binary(input_tensor: torch.Tensor, rank: int|None = None) -> torch.Tensor:
    """
    Simulates channel-wise 1-bit quantization using rank-k scale approximation.
    If rank is -1, uses channel-wise mean of absolute values as scale.
    Args:
        input_tensor: The input tensor (N, C).
        rank: The rank for scale approximation (-1 for mean).
    Returns:
        A tensor of the same size as the input, representing the dequantized result.
    """
    assert rank is not None, "Rank must be provided"
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"

    # NOTE: must use mean, otherwise the dequantized tensor's norm is too large, resulting nonsensical output
    if rank == -1:
        # Calculate channel-wise mean scale (reduction over N dimension)
        abs_input = torch.abs(input_tensor)
        chan_scale = torch.mean(abs_input, dim=0, keepdim=True) # Shape (1, C)
        tok_scale = torch.mean(abs_input, dim=1, keepdim=True) # Shape (N, 1)
        tok_scale = tok_scale / tok_scale.mean()
        scale = chan_scale * tok_scale
    else: # rank >= 1
        from xfuser.compact.compress_lowrank import svd, subspace_iter
        RANK=rank
        # subspace_iter input (N, C), output u_nk(N, K), v_kc(K, C)
        u_nk, v_kc, _ = subspace_iter(torch.abs(input_tensor), RANK, 2)
        scale = u_nk @ v_kc # (N, C)
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


def sim_int4(input_tensor: torch.Tensor, dim) -> torch.Tensor:
    """
    Simulates channel-wise INT4 quantization with zero-centered, min-max-based scaling.
    """
    assert input_tensor.dim() == 2, f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions."
    # assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    dtype = input_tensor.dtype
    input_float32 = input_tensor.float()
    val_max = torch.max(input_float32, dim=dim, keepdim=True)[0]
    val_min = torch.min(input_float32, dim=dim, keepdim=True)[0]
    scale = (val_max - val_min) / 15
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale) * 1e-8)
    zero_point_offset = torch.round(val_min / scale)

    # Quantize
    quantized_vals = torch.round((input_float32 - val_min) / scale)
    # Clamp to [0, 15] range first
    quantized_vals = torch.clamp(quantized_vals, min=0, max=15)
    # Shift to represent [-8, 7] if needed, but let's keep it [0, 15] for simplicity of dequant
    # Store as int8 as there's no int4 type
    dequantized_vals = quantized_vals * scale + val_min
    return dequantized_vals.to(dtype) # Return in original dtype


def sim_int2_minmax(input_tensor: torch.Tensor, dim) -> torch.Tensor:
    """
    Simulates channel-wise INT2 quantization with min-max-based scaling.
    """
    assert input_tensor.dim() == 2, f"Input tensor must be 2D, but got {input_tensor.dim()} dimensions."
    # assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    dtype = input_tensor.dtype
    input_float32 = input_tensor.float()
    val_max = torch.max(input_float32, dim=dim, keepdim=True)[0]
    val_min = torch.min(input_float32, dim=dim, keepdim=True)[0]
    scale = (val_max - val_min) / 3
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale) * 1e-8)
    zero_point_offset = torch.round(val_min / scale)

    # Quantize
    quantized_vals = torch.round((input_float32 - val_min) / scale)
    # Clamp to [0, 3] range first
    quantized_vals = torch.clamp(quantized_vals, min=0, max=3)
    # Shift to represent [-2, 1] if needed, but let's keep it [0, 3] for simplicity of dequant
    # Store as int8 as there's no int4 type
    dequantized_vals = quantized_vals * scale + val_min
    return dequantized_vals.to(dtype) # Return in original dtype