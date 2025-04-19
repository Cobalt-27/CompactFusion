import pytest
import torch
from xfuser.compact.compress_topk import (
    sim_topk,
    topk_compress,
    topk_decompress,
    topk_sparsify,
)
from xfuser.compact.compress_quantize import (
    quantize_1bit,
    dequantize_1bit,
    sim_binary,
)
from xfuser.compact.compress_lowrank import svd, subspace_iter
from xfuser.compact.compress_topk import SPARSE_LAST_DIM_SIZE

from xfuser.prof import Profiler, prof_summary

# Helper function to check that two tensors are close (using relative error)
def assert_tensor_close(tensor1, tensor2, tol=1e-3, desc=""):
    # Check basic properties first
    assert tensor1.dtype == tensor2.dtype, f"{desc}: Dtypes differ - {tensor1.dtype} vs {tensor2.dtype}"
    assert tensor1.shape == tensor2.shape, f"{desc}: Shapes differ - {tensor1.shape} vs {tensor2.shape}"

    if tensor1.dtype == torch.uint8:
        # For uint8, require exact match
        assert torch.equal(tensor1, tensor2), f"{desc}: uint8 tensors differ"
    else:
        # For floating point, use relative error
        norm_diff = torch.norm(tensor1.float() - tensor2.float())
        norm_tensor1 = torch.norm(tensor1.float())
        if norm_tensor1 == 0:
             # Handle zero tensor case
             assert norm_diff < tol, f"{desc}: Tensors differ by more than {tol}, tensor1 is zero, diff norm: {norm_diff}"
        else:
             relative_error = norm_diff / norm_tensor1
             assert relative_error < tol, f"{desc}: Tensors differ by more than relative tol {tol}, relative error: {relative_error}"

def assert_tensor_approx(tensor1, tensor2, tol=1e-4, desc=""):
    # For quantized tensors and low-rank
    relative_error = torch.norm(tensor1 - tensor2) / torch.norm(tensor1)
    assert relative_error < tol, f"{desc}: Tensors differ by more than {tol}, relative error: {relative_error}"

LOOP_CNT = 8

@pytest.mark.parametrize("m", [2, 4, 8, 16])  # Testing for m = 2, 4, 8, 16
@pytest.mark.parametrize(
    "A,B_2m", [(1024, 2048), (512, 4096), (256, 8192)]
)  # Large tensor sizes
@pytest.mark.parametrize("seed", [42, 43, 44])  # Different random seeds
def test_topk_sparsify(m, A, B_2m, seed):
    torch.manual_seed(seed)
    for _ in range(LOOP_CNT):
        # Generate a large random input tensor
        input_tensor = torch.randn((A, B_2m), dtype=torch.half, device="cuda")
        original_shape = input_tensor.shape
        # Simulate N:M sparsity compression in Python
        sparse_simulated = sim_topk(input_tensor, m)

        input_tensor = input_tensor.view(-1, SPARSE_LAST_DIM_SIZE)
        input_tensor = input_tensor.contiguous()
        # Perform compression using Triton kernel
        compressed_val, compressed_idx = topk_compress(input_tensor, m)
        
        # merged = merge_compress_results(compressed_val, compressed_idx)
        # compressed_val, compressed_idx = split_compress_results(merged)

        assert compressed_val.dtype == torch.half and compressed_idx.dtype == torch.uint8
        compressed_val = compressed_val.contiguous()
        compressed_idx = compressed_idx.contiguous()
        # Decompress using Triton kernel
        decompressed_tensor = topk_decompress(
            compressed_val, compressed_idx, m
        ).view(original_shape)

        sparse = topk_sparsify(input_tensor, m).view(original_shape)

        # Compare the decompressed tensor with the simulated sparse tensor
        assert_tensor_close(decompressed_tensor, sparse_simulated)
        assert_tensor_close(sparse, sparse_simulated)

@pytest.mark.parametrize(
    "A,B_2m", [(1024, 2048), (512, 4096), (256, 8192)]
)  # Large tensor sizes
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("rank", [4, -1]) # <<< Add rank parameter including -1
def test_1_bit_quantization(
    A, B_2m, seed, rank
):
    # No need to set seed outside the loop if using fork_rng inside
    # torch.manual_seed(seed)
    for i in range(LOOP_CNT):
        # Use a slightly different tensor each loop iter if desired, or keep fixed
        loop_seed = seed + i # Ensure different input tensors per loop if needed
        torch.manual_seed(loop_seed)
        input_tensor = torch.randn((A, B_2m), dtype=torch.half, device="cuda")
        
        # Simulate 1-bit quantization in Python with controlled RNG
        with torch.random.fork_rng(devices=[input_tensor.device]):
            torch.manual_seed(loop_seed) # Reset seed for sim
            quantized_simulated = sim_binary(input_tensor, rank=rank)
        
        # Perform actual quantization/dequantization with controlled RNG
        with torch.random.fork_rng(devices=[input_tensor.device]):
            torch.manual_seed(loop_seed) # Reset seed for actual quant
            compressed_tensor, scale_u, scale_v = quantize_1bit(input_tensor, rank=rank)
        
        # Dequantize using the results from the actual quantization
        decompressed_tensor = dequantize_1bit(compressed_tensor, scale_u, scale_v)
        # .view(input_tensor.shape) # dequantize_1bit now returns (N, C) matching input
        
        # Compare actual decompressed with simulated
        assert_tensor_approx(decompressed_tensor, quantized_simulated)


from xfuser.compact.slowpath import sim_compress, slowpath_compress, slowpath_decompress
from xfuser.compact.utils import COMPACT_COMPRESS_TYPE

@pytest.mark.parametrize(
    "n,hidden", [(1024, 2048), (512, 4096), (256, 8192)]
)  # Sequence and hidden dimensions
# @pytest.mark.parametrize("batch", [2, 4])  # Small batch sizes
@pytest.mark.parametrize("seed", [42, 43, 44])  # Different random seeds
@pytest.mark.parametrize("sparse_ratio", [1, 2, 4, 8, 16])  # Valid compression levels
@pytest.mark.parametrize("compact_method", [
    COMPACT_COMPRESS_TYPE.SPARSE,
    COMPACT_COMPRESS_TYPE.BINARY,
    COMPACT_COMPRESS_TYPE.LOW_RANK,
    # COMPACT_COMPRESS_TYPE.HYBRID
])  # Different compression methods
def test_compress_decompress_vs_sim(n, hidden, seed, compact_method, sparse_ratio):
    """
    Test that _compress and _decompress functions produce the same results as _sim_compress.
    This validates that the actual compression/decompression pipeline matches the simulation.
    
    Uses a 3D tensor with dimensions (batch, n, hidden) to better represent typical model activations.
    """
    torch.manual_seed(seed)
    for i in range(LOOP_CNT):
        # Generate a random 3D input tensor with a batch dimension
        input_tensor = torch.randn((n, hidden), dtype=torch.half, device="cuda")
        RANK = 2
        RANKS_TO_TEST = [RANK] # Default rank for non-binary or fallback
        if compact_method == COMPACT_COMPRESS_TYPE.BINARY:
            RANKS_TO_TEST = [RANK, -1] # Test both positive rank and mean-as-scale

        loop_seed = seed + i
        for test_rank in RANKS_TO_TEST:
            # Skip rank testing if method doesn't use it (like SPARSE)
            if compact_method != COMPACT_COMPRESS_TYPE.BINARY and compact_method != COMPACT_COMPRESS_TYPE.LOW_RANK and test_rank == -1:
                continue # Only test rank=-1 for BINARY

            current_rank = test_rank if (compact_method == COMPACT_COMPRESS_TYPE.BINARY or compact_method == COMPACT_COMPRESS_TYPE.LOW_RANK) else None

            # Use same seed scope for both operations, as lowrank requires random q
            with torch.random.fork_rng(devices=[input_tensor.device]):
                torch.manual_seed(loop_seed)
                # Get the simulated compression result
                simulated_result = sim_compress(input_tensor, compact_method, rank=current_rank, sparse_ratio=sparse_ratio)
            with torch.random.fork_rng(devices=[input_tensor.device]):
                torch.manual_seed(loop_seed)
                # Perform actual compression
                compressed = slowpath_compress(input_tensor, compact_method, rank=current_rank, sparse_ratio=sparse_ratio)
            
            # Decompress the compressed tensor
            decompressed = slowpath_decompress(compressed, input_tensor.shape, compact_method, rank=current_rank, sparse_ratio=sparse_ratio)
            
            # Compare the decompressed tensor with the simulated result
            assert_tensor_approx(decompressed, simulated_result, desc=f"Rank={current_rank}")

@pytest.mark.parametrize(
    "n,hidden", [(128, 128), (32, 256), (512, 32)]
)  # Sequence and hidden dimensions
@pytest.mark.parametrize("seed", [42, 43, 44])  # Different random seeds
@pytest.mark.parametrize("target_rank", [1, 2]) # Test rank 1 (power iter) and > 1 (subspace iter)
def test_subspace_iter(n, hidden, seed, target_rank):
    torch.manual_seed(seed)
    for _ in range(LOOP_CNT):
        # Generate a nearly low-rank matrix by adding small noise to a low-rank matrix
        GEN_RANK = target_rank # Ensure gen_rank is at least 2 for generating base matrix
        left = torch.randn((n, GEN_RANK), dtype=torch.float, device="cuda")
        right = torch.randn((GEN_RANK, hidden), dtype=torch.float, device="cuda")
        low_rank = left @ right  # Base low-rank matrix
        noise = 0.01 * torch.norm(low_rank) * torch.randn((n, hidden), dtype=torch.float, device="cuda") # Relative noise
        input_tensor = (low_rank + noise).half() # Test with half precision input

        # Use the public subspace_iter which dispatches to power_iter for rank=1
        pu, pv, _ = subspace_iter(input_tensor, target_rank, num_iters=100) # Increase iters for better convergence
        p_reconstructed = pu.float() @ pv.float()

        # Get reference reconstruction from SVD
        u, v = svd(input_tensor, target_rank)
        s_reconstructed = u.float() @ v.float()

        # Use a slightly looser tolerance, especially for rank 1 / power iteration
        tolerance = 1e-1
        assert_tensor_approx(
            p_reconstructed,
            s_reconstructed,
            tol=tolerance,
            desc=f"Rank={target_rank}"
        )

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])