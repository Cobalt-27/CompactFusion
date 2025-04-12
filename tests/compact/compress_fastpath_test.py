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
SHAPES = [(4096, 4096), (1024, 2048), (512, 8192)] # (CHANNEL, N_TOKENS)
SEEDS = [42, 43, 44]
UPDATE_CACHE_OPTIONS = [True, False]
RANKS_TO_TEST = [1, 4] # Test Rank 1 and Rank 4

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("update_cache", UPDATE_CACHE_OPTIONS)
@pytest.mark.parametrize("rank", RANKS_TO_TEST)
def test_binary_fastpath_e2e_vs_sim(shape, seed, update_cache, rank):
    """Compares the end-to-end binary fastpath (quant+dequant) kernel vs simulation for multiple ranks."""
    torch.manual_seed(seed)
    C, N = shape
    assert N % 8 == 0, "N_TOKENS must be divisible by 8"

    # --- Generate Inputs (C, N layout) ---
    x = torch.randn((C, N), dtype=torch.half, device="cuda").contiguous()
    base = (torch.randn_like(x) * 0.1).contiguous()

    # --- Quantization Step ---
    kernel_quant_args = (x, base, rank, update_cache)
    sim_quant_args = (x, base, rank, update_cache)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        packed_k, scale_u_k, scale_v_kn_k, new_base_k = binary_quant_fastpath(*kernel_quant_args)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        packed_s, scale_u_s, scale_v_kn_s, new_base_s = sim_binary_quant_fastpath(*sim_quant_args)

    assert_tensor_close(packed_k, packed_s, desc=f"Quant R{rank}: Packed")
    assert_tensor_close(scale_u_k, scale_u_s, desc=f"Quant R{rank}: Scale U(C,K)")
    assert_tensor_close(scale_v_kn_k, scale_v_kn_s, desc=f"Quant R{rank}: Scale V(K,N)")
    if update_cache:
        assert_tensor_close(new_base_k, new_base_s, desc=f"Quant R{rank}: New Base")
    else:
        assert new_base_k is None and new_base_s is None

    # --- Dequantization Step ---
    kernel_dequant_args = (packed_k, scale_u_k, scale_v_kn_k, base, rank)
    sim_dequant_args = (packed_k, scale_u_k, scale_v_kn_s, base, rank)

    recon_k = binary_dequant_fastpath(*kernel_dequant_args)

    recon_s = sim_binary_dequant_fastpath(*sim_dequant_args)

    assert_tensor_close(recon_k, recon_s, desc=f"Dequant R{rank}: Reconstructed")


if __name__ == "__main__":
    pytest.main([__file__])
