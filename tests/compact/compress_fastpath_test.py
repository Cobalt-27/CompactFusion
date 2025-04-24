import pytest
import torch
from xfuser.compact.fastpath import (
    binary_quant_fastpath,
    sim_binary_quant_fastpath,
    binary_dequant_fastpath,
    sim_binary_dequant_fastpath,
    int2_quant_fastpath,
    sim_int2_quant_fastpath,
    int2_dequant_fastpath,
    sim_int2_dequant_fastpath,
)
# Import helper from slowpath test
from tests.compact.compress_slowpath_test import assert_tensor_close, assert_tensor_approx

def assert_tensor_close_binary(a, b, desc="Binary Tensor", tol=1e-3):
    """Asserts that two integer tensors are close, allowing a small percentage of mismatch.
    
    Args:
        a, b: Integer tensors to compare
        desc: Description for error messages
        tol: Maximum allowed percentage of mismatches (e.g., 0.001 for 0.1% mismatch)
    """
    assert a.dtype == b.dtype, f"{desc}: Dtype mismatch {a.dtype} vs {b.dtype}"
    assert a.shape == b.shape, f"{desc}: Shape mismatch {a.shape} vs {b.shape}"
    if torch.equal(a, b):
        return # Exact match
    
    equal_count = torch.sum(a == b)
    total_elements = a.numel()
    mismatch_ratio = 1.0 - (equal_count / total_elements)
    
    assert mismatch_ratio <= tol, (
        f"{desc}: Mismatch ratio {mismatch_ratio:.6f} exceeds tolerance {tol}. "
        f"Differing elements: {total_elements - equal_count}"
    )

# Test parameters
SHAPES = [(4096, 4096), (2048, 1024), (8192, 512)] # (N_TOKENS, CHANNEL)
SEEDS = [42, 43, 44]
UPDATE_CACHE_OPTIONS = [True, False]
# Include rank=-1 for channel-wise mean testing
RANKS_TO_TEST = [-1, 1, 4] # Test Rank -1, Rank 1, and Rank 4

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("update_cache", UPDATE_CACHE_OPTIONS)
@pytest.mark.parametrize("rank", RANKS_TO_TEST)
def test_binary_fastpath_e2e_vs_sim(shape, seed, update_cache, rank):
    """Compares the end-to-end binary fastpath (quant+dequant) kernel vs simulation for multiple ranks including -1."""
    torch.set_float32_matmul_precision('high')# solve warning
    torch.manual_seed(seed)
    N, C = shape # Unpack as N, C
    assert C % 8 == 0, "CHANNEL must be divisible by 8"

    # --- Generate Inputs (N, C layout) ---
    x = torch.randn((N, C), dtype=torch.half, device="cuda").contiguous()
    base = (torch.randn_like(x) * 0.1).contiguous()

    # --- Quantization Step --- 
    kernel_quant_args = (x, base, rank, update_cache)
    sim_quant_args = (x, base, rank, update_cache)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        # Returns: packed(N,C//8), u(N,K), v(C,K), new_base(N,C)
        packed_k, scale_u_nk_k, scale_v_ck_k, new_base_k = binary_quant_fastpath(*kernel_quant_args)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        # Returns: packed(N,C//8), u(N,K), v(C,K), new_base(N,C)
        packed_s, scale_u_nk_s, scale_v_ck_s, new_base_s = sim_binary_quant_fastpath(*sim_quant_args)

    assert_tensor_close(packed_k, packed_s, desc=f"Quant R{rank}: Packed")
    # Check shapes consistent with rank and N,C layout
    expected_rank = rank if rank > 0 else 1
    assert packed_k.shape == (N, C // 8)
    assert packed_s.shape == (N, C // 8)
    assert scale_u_nk_k.shape == (N, expected_rank)
    assert scale_u_nk_s.shape == (N, expected_rank)
    assert scale_v_ck_k.shape == (C, expected_rank)
    assert scale_v_ck_s.shape == (C, expected_rank)

    assert_tensor_close(scale_u_nk_k, scale_u_nk_s, desc=f"Quant R{rank}: Scale U (N,K)")
    assert_tensor_close(scale_v_ck_k, scale_v_ck_s, desc=f"Quant R{rank}: Scale V (C,K)")
    if update_cache:
        assert new_base_k is not None and new_base_s is not None
        assert_tensor_close(new_base_k, new_base_s, desc=f"Quant R{rank}: New Base (N,C)")
    else:
        assert new_base_k is None and new_base_s is None

    # --- Dequantization Step (Rank is now inferred from scales) ---
    # Args: packed(N,C//8), u(N,K), v(C,K), base(N,C)
    kernel_dequant_args = (packed_k, scale_u_nk_k, scale_v_ck_k, base)
    sim_dequant_args = (packed_s, scale_u_nk_s, scale_v_ck_s, base) # Use sim scales for sim dequant

    recon_k = binary_dequant_fastpath(*kernel_dequant_args)

    recon_s = sim_binary_dequant_fastpath(*sim_dequant_args)

    assert_tensor_close(recon_k, recon_s, desc=f"Dequant R{rank}: Reconstructed (N,C)")
    
INT2_FASTPATH_TOL = 0.02

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("update_cache", UPDATE_CACHE_OPTIONS)
def test_int2_fastpath_e2e_vs_sim(shape, seed, update_cache):
    """Compares the end-to-end INT2 fastpath (quant+dequant) kernel vs simulation."""
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed)
    N, C = shape # Unpack as N, C
    assert C % 4 == 0, "CHANNEL must be divisible by 4 for INT2 packing"
    # --- Generate Inputs (N, C layout) --- 
    x = torch.randn((N, C), dtype=torch.half, device="cuda").contiguous()
    base = (torch.randn_like(x) * 0.1).contiguous()

    # --- Quantization Step --- 
    # Add rank=-1, although it's default and enforced inside
    kernel_quant_args = (x, base, update_cache, -1) 
    sim_quant_args = (x, base, update_cache, -1)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        # Returns: packed(N,C//4), scale_u(N,1), scale_v(C,1), new_base(N,C)|None
        packed_k, scale_u_k, scale_v_k, new_base_k = int2_quant_fastpath(*kernel_quant_args)
    
    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        # Returns: packed(N,C//4), scale_u(N,1), scale_v(C,1), new_base(N,C)|None
        packed_s, scale_u_s, scale_v_s, new_base_s = sim_int2_quant_fastpath(*sim_quant_args)

    # Use binary comparison for packed data (allow 0.1% mismatch)
    assert_tensor_close_binary(packed_k, packed_s, desc="Quant INT2: Packed", tol=1e-3)
    assert packed_k.shape == (N, C // 4)
    assert packed_s.shape == (N, C // 4)

    # Use standard float comparison for scales and base
    # Check shapes match U(N,1), V(C,1)
    assert scale_u_k.shape == (N, 1)
    assert scale_u_s.shape == (N, 1)
    assert scale_v_k.shape == (C, 1)
    assert scale_v_s.shape == (C, 1)
    # Compare scales U and V
    assert_tensor_approx(scale_u_k, scale_u_s, desc="Quant INT2: Scale U (N,1)", tol=INT2_FASTPATH_TOL)
    assert_tensor_approx(scale_v_k, scale_v_s, desc="Quant INT2: Scale V (C,1)", tol=INT2_FASTPATH_TOL)
    
    if update_cache:
        assert new_base_k is not None and new_base_s is not None
        assert_tensor_approx(new_base_k, new_base_s, desc="Quant INT2: New Base (N,C)", tol=INT2_FASTPATH_TOL)
    else:
        assert new_base_k is None and new_base_s is None
        
    # --- Dequantization Step --- 
    # Args: packed(N,C//4), scale_u(N,1), scale_v(C,1), base(N,C)
    kernel_dequant_args = (packed_k, scale_u_k, scale_v_k, base)
    sim_dequant_args = (packed_s, scale_u_s, scale_v_s, base)

    recon_k = int2_dequant_fastpath(*kernel_dequant_args)
    recon_s = sim_int2_dequant_fastpath(*sim_dequant_args)

    assert_tensor_approx(recon_k, recon_s, desc="Dequant INT2: Reconstructed (N,C)", tol=INT2_FASTPATH_TOL)


if __name__ == "__main__":
    pytest.main([__file__])
