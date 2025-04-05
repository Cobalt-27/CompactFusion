import triton
import triton.language as tl
import torch
from pipefuser.prof import Profiler

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
    grid = (A,)
    with Profiler.scope("compact.quantize_1bit_kernel"):
        _quantize_1bit_kernel[grid](
            input_ptr=input_tensor,
            output_ptr=output_tensor,
            B=B,
        )
    return output_tensor, scale

@triton.jit
def _quantize_1bit_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr
):
    """
    Quantize input tensor to 1-bit and pack 8 values into one int8.
    Uses channel-wise scale and sign-based quantization.
    
    Args:
        input_ptr: Pointer to input tensor (FP16) [A, B]
        output_ptr: Pointer to output tensor (INT8) - packed binary values [A, B / 8]
        B: Number of blocks per row
    """
    row_id = tl.program_id(0)
    block_indices = tl.arange(0, B)  # Shape: [B]
    block_starts = row_id * B * 8 + block_indices * 8  # Shape: [B]
    output_starts = row_id * B + block_indices
    packed = tl.zeros((B,), dtype=tl.uint8)  # Shape: [B]
    for i in range(8):
        offsets = block_starts + i
        x = tl.load(input_ptr + offsets)  # Shape: [B]
        binary = (x >= 0).to(tl.uint8)  # Shape: [B]
        packed = (packed | (binary << i)).to(tl.uint8)
    
    tl.store(output_ptr + output_starts, packed)

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
    output = torch.empty((A, B*8), dtype=torch.half, device=input_tensor.device)
    grid = (A,)
    with Profiler.scope("compact.dequantize_1bit_kernel"):
        _dequantize_1bit_kernel[grid](
            input_ptr=input_tensor,
            output_ptr=output,
            scale_ptr=scale,
            B=B,
        )
    output = torch.transpose(output, 0, 1).contiguous()
    return output 

@triton.jit
def _dequantize_1bit_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    B: tl.constexpr,
):
    """
    Dequantize packed 1-bit values back to FP16.
    Uses channel-wise scale and sign-based dequantization.
    
    Args:
        input_ptr: Pointer to input tensor (INT8) - packed binary values [A, B / 8]
        output_ptr: Pointer to output tensor (FP16) [A, B]
        scale_ptr: Pointer to channel-wise quantization scale (FP16) [A]
        B: Size of the block to process
    """
    row_id = tl.program_id(0) 
    row_start = row_id * B  # Total elements per row: num_blocks * B
    row_output_start = row_id * B * 8
    block_indices = tl.arange(0, B)  # Shape: [B]
    block_starts = row_start + block_indices  # Shape: [B]
    output_starts = row_output_start + block_indices * 8
    scale = tl.load(scale_ptr + row_id)  # Shape: [1]
    packed = tl.load(input_ptr + block_starts)  # Shape: [B]
    for i in range(8):
        bits = (packed >> i) & 1  # Shape: [B]
        scaled = (2 * bits - 1) * scale  # Shape: [B]
        tl.store(output_ptr + output_starts + i, scaled)

@torch.compile
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
    scale = torch.mean(torch.abs(input_tensor), dim=tuple(range(input_tensor.ndim - 1)), keepdim=True) if scale is None else scale
    # from pipefuser.compact.compress_lowrank import svd, subspace_iter
    # # u, v = svd(torch.abs(input_tensor), 2)
    # u, v = subspace_iter(torch.abs(input_tensor), 2, 2)
    # scale = u @ v
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