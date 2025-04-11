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
DELTA_DECAY = 0.5
RESIDUAL_LEVELS = [1, 2]
RANKS_TO_TEST = [1, 4] # <<< Test Rank 1 and Rank 4

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("update_cache", UPDATE_CACHE_OPTIONS)
@pytest.mark.parametrize("residual_level", RESIDUAL_LEVELS)
@pytest.mark.parametrize("rank", RANKS_TO_TEST) # <<< Add rank parameterization
def test_binary_fastpath_e2e_vs_sim(shape, seed, update_cache, residual_level, rank):
    """Compares the end-to-end binary fastpath (quant+dequant) kernel vs simulation for both residual levels and multiple ranks."""
    torch.manual_seed(seed)
    C, N = shape
    assert N % 8 == 0, "N_TOKENS must be divisible by 8"

    # --- Generate Inputs (C, N layout) ---
    x = torch.randn((C, N), dtype=torch.half, device="cuda").contiguous()
    base = (torch.randn_like(x) * 0.1).contiguous()
    delta_base = (torch.randn_like(x) * 0.05).contiguous() if residual_level == 2 else None
    kernel_delta_base_arg = delta_base

    # --- Quantization Step ---
    kernel_quant_args = (x, base, kernel_delta_base_arg, rank, update_cache, DELTA_DECAY, residual_level) # <<< Added rank
    sim_quant_args = (x, base, delta_base, rank, update_cache, DELTA_DECAY, residual_level) # <<< Added rank

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        packed_k, scale_u_k, scale_v_kn_k, new_base_k, new_db_k = binary_quant_fastpath(*kernel_quant_args)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(seed)
        packed_s, scale_u_s, scale_v_kn_s, new_base_s, new_db_s = sim_binary_quant_fastpath(*sim_quant_args)

    assert_tensor_close(packed_k, packed_s, desc=f"Quant L{residual_level} R{rank}: Packed")
    assert_tensor_close(scale_u_k, scale_u_s, desc=f"Quant L{residual_level} R{rank}: Scale U(C,K)")
    assert_tensor_close(scale_v_kn_k, scale_v_kn_s, desc=f"Quant L{residual_level} R{rank}: Scale V(K,N)")
    if update_cache:
        assert_tensor_close(new_base_k, new_base_s, desc=f"Quant L{residual_level} R{rank}: New Base")
        if residual_level == 2:
            assert_tensor_close(new_db_k, new_db_s, desc=f"Quant L{residual_level} R{rank}: New Delta Base")
        else:
            assert new_db_k is None, f"Quant L{residual_level} R{rank}: Kernel new_db should be None"
            assert new_db_s is None, f"Quant L{residual_level} R{rank}: Sim new_db should be None"
    else:
        assert new_base_k is None and new_base_s is None
        assert new_db_k is None and new_db_s is None

    # --- Dequantization Step ---
    kernel_dequant_args = (packed_k, scale_u_k, scale_v_kn_k, base, kernel_delta_base_arg, rank, DELTA_DECAY, residual_level)
    sim_dequant_args = (packed_k, scale_u_k, scale_v_kn_s, base, delta_base, rank, DELTA_DECAY, residual_level)

    recon_k, new_db_dequant_k = binary_dequant_fastpath(*kernel_dequant_args)

    recon_s, new_db_dequant_s = sim_binary_dequant_fastpath(*sim_dequant_args)

    assert_tensor_close(recon_k, recon_s, desc=f"Dequant L{residual_level} R{rank}: Reconstructed")
    if residual_level == 2:
        assert_tensor_close(new_db_dequant_k, new_db_dequant_s, desc=f"Dequant L{residual_level} R{rank}: New Delta Base")
    else:
        assert new_db_dequant_k is None, f"Dequant L{residual_level} R{rank}: Kernel new_db should be None"
        assert new_db_dequant_s is None, f"Dequant L{residual_level} R{rank}: Sim new_db should be None"


if __name__ == "__main__":
    pytest.main([__file__])
