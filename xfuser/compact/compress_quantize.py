import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler

def quantize_1bit(
    input_tensor: torch.Tensor,
):
    """
    Quantize input tensor to 1-bit and pack 8 values into one uint8.
    Uses channel-wise scale and sign-based quantization.
    
    Args:
        input_tensor: Input tensor (FP16), shape (n_tokens, channel)
        
    Returns:
        Tuple of (quantized_tensor, scale):
            - quantized_tensor: Packed tensor (INT8) containing binary values
            - scale: Channel-wise quantization scale (FP16)
    """
    assert input_tensor.dtype == torch.half, "Input tensor must be FP16"
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    # Each row has B * 8 and the scale is calculated over the row, 
    # so the scale is a 1D tensor with shape (B,)
    scale = torch.mean(torch.abs(input_tensor), dim=0).flatten()
    # transpose to (channel, n_tokens) for better cache performance
    input_tensor = torch.transpose(input_tensor, 0, 1).contiguous()
    A, B_8 = input_tensor.shape
    assert B_8 % 8 == 0, "B_8 must be divisible by 8"
    
    B = B_8 // 8
    output_tensor = input_tensor.new_empty((A, B), dtype=torch.uint8).contiguous()
    # Define the block size for the packed dimension (must match the kernel's constexpr)
    BLOCK_SIZE_B_PACKED = 256
    # Calculate the grid size
    # Grid dim 0 is for channels (A)
    # Grid dim 1 needs to cover the packed dimension (B) in blocks of BLOCK_SIZE_B_PACKED
    grid = (A, triton.cdiv(B, BLOCK_SIZE_B_PACKED))
    with Profiler.scope("compact.quantize_1bit_kernel"):
        _quantize_1bit_kernel[grid](
            input_ptr=input_tensor,
            output_ptr=output_tensor,
            A=A,
            B_8=B_8,
            # Pass the block size used in grid calculation
            BLOCK_SIZE_B_PACKED=BLOCK_SIZE_B_PACKED, 
        )
    return output_tensor, scale

@triton.jit
def _quantize_1bit_kernel(
    input_ptr,
    output_ptr,
    A,
    B_8,
    BLOCK_SIZE_B_PACKED: tl.constexpr = 32, # Number of uint8 elements to process per instance
):
    """
    Quantize input tensor to 1-bit and pack 8 values into one uint8.
    Uses channel-wise scale and sign-based quantization.
    
    Args:
        input_ptr: Input tensor (FP16), shape (channel, n_tokens)
        output_ptr: Output tensor (INT8), shape (channel, n_tokens // 8)
        A: Number of channels
        B_8: Number of tokens (must be divisible by 8)
        BLOCK_SIZE_B_PACKED: Block size for the packed B dimension (number of uint8 outputs)
    """
    # Get program ID for the channel dimension (A)
    pid_a = tl.program_id(0)
    # Get program ID for the packed token dimension (B = B_8 // 8)
    pid_b_packed = tl.program_id(1)

    # Calculate actual B (number of packed uint8 values per channel)
    B_packed = B_8 // 8

    # Calculate the starting offset for the current block in the packed dimension
    b_packed_offset = pid_b_packed * BLOCK_SIZE_B_PACKED
    # Create offsets for the packed output dimension within the current block
    offs_b_packed = b_packed_offset + tl.arange(0, BLOCK_SIZE_B_PACKED)
    # Create a mask to handle blocks that might go beyond the actual packed dimension size
    mask_b_packed = offs_b_packed < B_packed

    # Calculate the starting offset for the input tensor (corresponding to the packed block)
    # Each packed value corresponds to 8 input values
    b_8_offset = b_packed_offset * 8
    
    # Initialize the packed uint8 result for the current block
    packed_result = tl.zeros((BLOCK_SIZE_B_PACKED,), dtype=tl.uint8)

    # Iterate through the 8 bits for packing
    for i in range(8):
        # Calculate offsets for the input tensor (FP16 values)
        # We need BLOCK_SIZE_B_PACKED elements for this bit position 'i' across the block
        offs_b_8_for_bit_i = b_8_offset + tl.arange(0, BLOCK_SIZE_B_PACKED) * 8 + i
        
        # Load input values corresponding to the current bit 'i' for the entire block
        input_offsets = pid_a * B_8 + offs_b_8_for_bit_i
        # The mask should ensure we only load valid elements within B_8 for this channel
        # And also respect the boundary of the current *packed* block
        load_mask = mask_b_packed # This ensures we only consider elements within the valid packed block range
        
        x = tl.load(
            input_ptr + input_offsets,
            mask=load_mask,
            other=0.0, # Use 0.0 for values outside the valid range
        )
        
        # Determine the binary value (1 if x > 0, else 0)
        # Apply the mask again to ensure we only set bits for valid elements
        binary_value = tl.where(x >= 0, 1, 0).to(tl.uint8) & tl.where(load_mask, 1, 0).to(tl.uint8)
        
        # Shift the binary value to the correct bit position 'i' and OR it with the result
        # Ensure all operations maintain uint8 type
        shifted_binary = (binary_value << i).to(tl.uint8)
        packed_result = (packed_result | shifted_binary).to(tl.uint8)

    # Calculate the pointer for the output tensor (packed uint8 values)
    output_ptr_base = output_ptr + pid_a * B_packed
    output_offsets = output_ptr_base + offs_b_packed
    
    # Store the packed result block
    tl.store(
        output_offsets,
        packed_result,
        mask=mask_b_packed, # Mask ensures we only write within the valid packed dimension
    )

def dequantize_1bit(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Dequantize packed 1-bit values back to FP16.
    Uses channel-wise scale and sign-based dequantization.
    
    Args:
        input_tensor: Quantized tensor (UINT8) - packed binary values
        scale: Channel-wise quantization scale (FP16)
        
    Returns:
        Dequantized tensor (FP16)
    """
    assert input_tensor.dtype == torch.uint8, "Input tensor must be UINT8"
    assert scale.dtype == torch.half, "Scale must be FP16"
    assert scale.ndim == 1, "Scale must be a 1D tensor"
    assert input_tensor.ndim == 2, "Input tensor must be 2D"
    input_tensor = input_tensor.contiguous()
    (A, B) = input_tensor.shape
    B_8 = B * 8 # Calculate the unpacked dimension size
    output = torch.empty((A, B_8), dtype=torch.half, device=input_tensor.device)
    # Define the block size (must match the kernel's constexpr)
    BLOCK_SIZE_B = 256
    # Calculate the 2D grid
    # Grid dim 0 is for channels (A)
    # Grid dim 1 needs to cover the packed dimension (B) in blocks of BLOCK_SIZE_B
    grid = (A, triton.cdiv(B, BLOCK_SIZE_B))
    with Profiler.scope("compact.dequantize_1bit_kernel"):
        _dequantize_1bit_kernel[grid](
            input_ptr=input_tensor,
            output_ptr=output,
            scale_ptr=scale,
            A=A,
            B=B,
            B_8=B_8,
            BLOCK_SIZE_B=BLOCK_SIZE_B,
        )
    output = torch.transpose(output, 0, 1).contiguous()
    return output

@triton.jit
def _dequantize_1bit_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    A,  # Number of channels
    B,  # Number of packed elements per channel (uint8)
    B_8, # Number of unpacked elements per channel (fp16)
    BLOCK_SIZE_B: tl.constexpr = 256, # Block size for the packed dimension
):
    """
    Dequantize packed 1-bit values back to FP16 using a blocked approach.
    Uses channel-wise scale and sign-based dequantization.
    
    Args:
        input_ptr: Pointer to input tensor (UINT8) - packed binary values [A, B]
        output_ptr: Pointer to output tensor (FP16) [A, B_8]
        scale_ptr: Pointer to channel-wise quantization scale (FP16) [A]
        A: Number of channels.
        B: Number of packed uint8 elements per channel.
        B_8: Number of unpacked fp16 elements per channel (B * 8).
        BLOCK_SIZE_B: Processing block size for the packed dimension B.
    """
    # Program IDs
    pid_a = tl.program_id(0)  # Channel ID
    pid_b = tl.program_id(1)  # Block ID in the packed dimension B

    # Calculate offsets for the current block in the packed dimension
    b_offset = pid_b * BLOCK_SIZE_B
    offs_b = b_offset + tl.arange(0, BLOCK_SIZE_B)
    mask_b = offs_b < B  # Mask for packed dimension elements

    # Load scale for the current channel
    scale = tl.load(scale_ptr + pid_a)

    # Calculate pointer offsets for loading packed data
    packed_ptrs = input_ptr + pid_a * B + offs_b
    # Load packed data for the current block, masking invalid elements
    packed_data = tl.load(packed_ptrs, mask=mask_b, other=0) # Load 0 for masked elements

    # Calculate base offset for output pointer (start of the unpacked row)
    output_row_start_ptr = output_ptr + pid_a * B_8

    # Unpack 8 bits
    for i in range(8):
        # Extract the i-th bit from each packed uint8 value
        # packed_data has shape [BLOCK_SIZE_B]
        bits = ((packed_data >> i) & 1).to(tl.int8) # Result shape: [BLOCK_SIZE_B]
        
        # Dequantize: 0 -> -scale, 1 -> +scale
        # Simple approach: if bit is 1, use +scale, otherwise use -scale
        scaled = tl.where(bits == 1, scale, -scale)

        # Calculate output pointers for the current bit position
        # Each packed element at offs_b corresponds to 8 output elements starting at offs_b * 8
        offs_b_8 = offs_b * 8 + i # Offset within the unpacked row for the i-th bit
        output_ptrs = output_row_start_ptr + offs_b_8
        
        # Store the dequantized values, masking invalid elements
        # The mask_b ensures we only write for valid packed elements from the input block
        tl.store(output_ptrs, scaled, mask=mask_b)

# @torch.compile
def sim_binary(input_tensor: torch.Tensor, scale: torch.Tensor = None) -> torch.Tensor:
    """
    Simulates channel-wise 1-bit quantization with per-channel scale.
    This is a simulation function for the BINARY compression type.

    Args:
        input_tensor: The input tensor. The last dimension is assumed to be the channel dimension.

    Returns:
        A tensor of the same size as the input, representing the dequantized result.
    """
    # NOTE: must use mean, otherwise the dequantized tensor's norm is too large, resulting nonsensical output
    # scale = torch.mean(torch.abs(input_tensor), dim=tuple(range(input_tensor.ndim - 1)), keepdim=True) if scale is None else scale
    from xfuser.compact.compress_lowrank import svd, subspace_iter
    # # u, v = svd(torch.abs(input_tensor), 2)
    u, v, _ = subspace_iter(torch.abs(input_tensor), 2, 2)
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