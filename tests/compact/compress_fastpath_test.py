import pytest
import torch
from xfuser.compact.fastpath import (
    binary_quant_fastpath,
    sim_binary_quant_fastpath,
    binary_dequant_fastpath,
    sim_binary_dequant_fastpath,
)
# Import helper from slowpath test
from tests.compact.compress_slowpath_test import assert_tensor_close

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


if __name__ == "__main__":
    pytest.main([__file__])
