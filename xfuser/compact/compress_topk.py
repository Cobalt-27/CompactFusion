import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler

# NOTE: This topk implementation is obsolete and not used in the current codebase.

SPARSE_LAST_DIM_SIZE = 1024
VALID_COMPRESS_LEVELS = [1, 2, 4, 8, 16]
# @Profiler.prof_func("compact.compress")
def topk_compress(input_tensor: torch.Tensor, m: int):
    """
    Compresses the input tensor using the 1:M sparsity method.
    Each idx byte contains two 4-bit indices (high 4 bits for block1, low 4 bits for block2).
    """
    # Get the input tensor dimensions
    A, B_2m = input_tensor.shape
    # Compute the number of blocks per row
    B = B_2m // (2 * m)
    assert B_2m % (2 * m) == 0, "The number of columns must be divisible by 2*m."

    # Output tensor shapes
    output_val_shape = (A, B * 2)
    output_idx_shape = (A, B)

    # Create output tensors
    output_val_tensor = input_tensor.new_empty(output_val_shape, dtype=torch.half)
    output_idx_tensor = input_tensor.new_empty(output_idx_shape, dtype=torch.uint8)

    # Launch Triton kernel
    grid = (A,)  # One thread per row
    with Profiler.scope("compact.compress_kernel"):
        _compress_kernel[grid](
            input_ptr=input_tensor,
            output_val_ptr=output_val_tensor,
            output_idx_ptr=output_idx_tensor,
            m=m,
            B=B,  # Pass as tl.constexpr
        )

    return output_val_tensor, output_idx_tensor

@triton.jit
def _compress_kernel(
    input_ptr, output_val_ptr, output_idx_ptr, m: tl.constexpr, B: tl.constexpr
):
    """
    compression kernel for N:M sparsity with N = 1.
    """
    row_id = tl.program_id(0)  # Each program processes one row

    # Compute the starting offset for this row
    row_start = row_id * B * 2 * m  # Total elements per row: num_blocks * 2 * m

    # Block indices (vectorized over all blocks in the row)
    block_indices = tl.arange(0, B)  # Shape: [num_blocks]

    # Compute starting offsets for block1 and block2
    block1_starts = row_start + block_indices * 2 * m  # Shape: [num_blocks]
    block2_starts = block1_starts + m  # Shape: [num_blocks]

    # Element indices within each block
    element_indices = tl.arange(0, m)  # Shape: [m]

    # Compute offsets for loading elements in block1 and block2
    offsets1 = (
        block1_starts[:, None] + element_indices[None, :]
    )  # Shape: [num_blocks, m]
    offsets2 = (
        block2_starts[:, None] + element_indices[None, :]
    )  # Shape: [num_blocks, m]

    # Load data for block1 and block2
    x1 = tl.load(input_ptr + offsets1)  # Shape: [num_blocks, m]
    x2 = tl.load(input_ptr + offsets2)  # Shape: [num_blocks, m]

    # Compute absolute values
    abs_x1 = tl.abs(x1)  # Shape: [num_blocks, m]
    abs_x2 = tl.abs(x2)  # Shape: [num_blocks, m]

    # Find indices of maximum absolute values
    max_idx1 = tl.argmax(abs_x1, axis=1)  # Shape: [num_blocks]
    max_idx2 = tl.argmax(abs_x2, axis=1)  # Shape: [num_blocks]

    # Compute the maximum values' offsets
    max_val1_offsets = block1_starts + max_idx1  # Shape: [num_blocks]
    max_val2_offsets = block2_starts + max_idx2  # Shape: [num_blocks]

    # Load the maximum values from input_ptr
    val1 = tl.load(input_ptr + max_val1_offsets)  # Shape: [num_blocks]
    val2 = tl.load(input_ptr + max_val2_offsets)  # Shape: [num_blocks]

    # Store the values
    val_offsets = row_id * B * 2 + block_indices * 2  # Shape: [num_blocks]
    tl.store(output_val_ptr + val_offsets + 0, val1)
    tl.store(output_val_ptr + val_offsets + 1, val2)

    # Pack the indices (high 4 bits for block1, low 4 bits for block2)
    max_idx1_int8 = max_idx1.to(tl.uint8)
    max_idx2_int8 = max_idx2.to(tl.uint8)
    packed_idx = (max_idx1_int8 << 4) | max_idx2_int8  # Shape: [num_blocks]

    idx_offsets = row_id * B + block_indices  # Shape: [num_blocks]
    tl.store(output_idx_ptr + idx_offsets, packed_idx)


# @Profiler.prof_func("compact.decompress")
def topk_decompress(
    compressed_val_tensor: torch.Tensor, compressed_idx_tensor: torch.Tensor, m: int
):
    A, B = compressed_idx_tensor.shape
    # Decompressed output tensor shape: [A, 2*m*B] (each row has 2*m*B elements)
    output_shape = (A, 2 * m * B)
    output_tensor = compressed_val_tensor.new_zeros(output_shape)
    # Launch Triton kernel
    grid = (A,)  # One thread per row
    with Profiler.scope("compact.decompress_kernel"):
        _decompress_kernel[grid](
            compressed_val_tensor,
            compressed_idx_tensor,
            output_tensor,
            m=m,  # Pass as tl.constexpr
            B=B,  # Pass as tl.constexpr
        )
    return output_tensor

@triton.jit
def _decompress_kernel(
    compressed_val_ptr, compressed_idx_ptr, output_ptr, m: tl.constexpr, B: tl.constexpr
):
    row_id = tl.program_id(0)  # Each program processes one row

    # Compute the starting offsets
    row_val_start = row_id * 2 * B  # Each row has 2*B values (2 values per block)
    row_idx_start = row_id * B  # Each row has B packed indices
    row_output_start = (
        row_id * 2 * m * B
    )  # Position to start storing decompressed values

    # Block indices
    block_indices = tl.arange(0, B)  # Shape: [B]

    # Load packed indices
    packed_idx = tl.load(
        compressed_idx_ptr + row_idx_start + block_indices
    )  # Shape: [B]

    # Unpack indices
    max_idx1 = ((packed_idx >> 4) & 0xF).to(tl.int32)  # Shape: [B]
    max_idx2 = (packed_idx & 0xF).to(tl.int32)  # Shape: [B]

    # Load compressed values
    val_indices = row_val_start + block_indices * 2  # Shape: [B]
    max_val1 = tl.load(compressed_val_ptr + val_indices)  # Shape: [B]
    max_val2 = tl.load(compressed_val_ptr + val_indices + 1)  # Shape: [B]

    # Compute output positions for block1 and block2
    positions1 = row_output_start + block_indices * 2 * m + max_idx1  # Shape: [B]
    positions2 = row_output_start + block_indices * 2 * m + m + max_idx2  # Shape: [B]

    # Store the maximum values directly into the output tensor at the correct positions
    tl.store(output_ptr + positions1, max_val1)
    tl.store(output_ptr + positions2, max_val2)

def topk_sparsify(input_tensor: torch.Tensor, m: int):
    """
    Perform end-to-end N:M sparsification on the input tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor to sparsify. Must be 2D and on GPU.
        m (int): Sparsity ratio (N:M, where N = 1 and M = `m`).

    Returns:
        torch.Tensor: Sparsified output tensor with the same shape as the input.
    """
    # Ensure the input is valid
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    assert input_tensor.dim() == 2, "Input tensor must be 2D."
    A, B_m = input_tensor.shape
    assert B_m % m == 0, "The number of columns must be divisible by m."
    # Compute the number of blocks per row
    B = B_m // m
    # Output tensor shape
    output_tensor = torch.zeros_like(input_tensor)
    # Launch Triton kernel
    grid = (A,)  # One thread per row
    _sparsify_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        m=m,
        B=B,  # Pass as tl.constexpr
    )
    return output_tensor

@triton.jit
def _sparsify_kernel(input_ptr, output_ptr, m: tl.constexpr, B: tl.constexpr):
    row_id = tl.program_id(0)  # Each program processes one row
    # Starting offset for the row
    row_start = row_id * B * m  # Total elements per row: B * m
    # Block indices (vectorized over all blocks in the row)
    block_indices = tl.arange(0, B)  # Shape: [B]
    # Compute starting offsets for each block
    block_starts = row_start + block_indices * m  # Shape: [B]
    # Element indices within each block
    element_indices = tl.arange(0, m)  # Shape: [m]
    # Compute offsets for loading elements in each block
    offsets = block_starts[:, None] + element_indices[None, :]  # Shape: [B, m]
    # Load the data
    x = tl.load(input_ptr + offsets)  # Shape: [B, m]
    # Compute absolute values
    abs_x = tl.abs(x)  # Shape: [B, m]
    # Find the index of the maximum absolute value in each block
    max_idx = tl.argmax(abs_x, axis=1)  # Shape: [B]
    # Generate masks for keeping only the maximum values
    mask = element_indices[None, :] == max_idx[:, None]  # Shape: [B, m]
    # Apply the mask to the input values
    sparsified = tl.where(mask, x, 0.0)  # Shape: [B, m]
    # Store the sparsified values back to the output tensor
    tl.store(output_ptr + offsets, sparsified)

def sim_topk(x: torch.Tensor, m: int):
    """
    N:M sparsity with N = 1.
    Simulate the compression&decompression process, not actually reducing the size.
    Return a sparse tensor, keeping the n elements with largest abs in every m elements.
    """
    N = 1
    assert x.shape[-1] % m == 0
    original_shape = x.shape
    x = x.view(-1, m)
    values, indices = torch.topk(x.abs(), N)
    x_sparse = torch.zeros_like(x)
    x_sparse.scatter_(dim=-1, index=indices, src=x.gather(dim=-1, index=indices))
    x_sparse = x_sparse.view(original_shape)
    assert x_sparse.count_nonzero() <= N * x.numel() // m
    return x_sparse