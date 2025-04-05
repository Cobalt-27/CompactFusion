import pytest
import torch
from xfuser.compact.fastpath import (
    binary_quant_fastpath,
    sim_binary_quant_fastpath,
    binary_dequant_fastpath,
    sim_binary_dequant_fastpath
)
# Import helper from slowpath test
from tests.compact.compress_slowpath_test import assert_tensor_close

# Test parameters
SHAPES = [(4096, 4096), (1024, 2048), (512, 8192)] # (CHANNEL, N_TOKENS)
SEEDS = [42, 43, 44]
UPDATE_CACHE_OPTIONS = [True, False]
DELTA_DECAY = 0.5

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("update_cache", UPDATE_CACHE_OPTIONS)
def test_binary_fastpath_e2e_vs_sim(shape, seed, update_cache):
    """Compares the end-to-end binary fastpath (quant+dequant) kernel vs simulation."""
    torch.manual_seed(seed)
    C, N = shape
    assert N % 8 == 0, "N_TOKENS must be divisible by 8"

    # --- Generate Inputs (C, N layout) ---
    x = torch.randn((C, N), dtype=torch.half, device="cuda").contiguous()
    base = (torch.randn_like(x) * 0.1).contiguous()
    delta_base = (torch.randn_like(x) * 0.05).contiguous()

    # --- Quantization Step --- 
    # Run kernel quant
    packed_k, scale_k, new_base_k, new_db_k = binary_quant_fastpath(
        x, base, delta_base, update_cache=update_cache, delta_decay_factor=DELTA_DECAY
    )

    # Run simulation quant
    packed_s, scale_s, new_base_s, new_db_s = sim_binary_quant_fastpath(
        x, base, delta_base, update_cache=update_cache, delta_decay_factor=DELTA_DECAY
    )

    # Compare quant results
    assert_tensor_close(packed_k, packed_s, desc="Quant: Packed DD")
    assert_tensor_close(scale_k, scale_s, desc="Quant: Scale DD")
    if update_cache:
        assert_tensor_close(new_base_k, new_base_s, desc="Quant: New Base")
        assert_tensor_close(new_db_k, new_db_s, desc="Quant: New Delta Base")
    else:
        assert new_base_k is None and new_base_s is None
        assert new_db_k is None and new_db_s is None

    # --- Dequantization Step (using kernel's quant output) --- 
    # Run kernel dequant
    recon_k, new_db_dequant_k = binary_dequant_fastpath(
        packed_k, scale_k, base, delta_base, delta_decay_factor=DELTA_DECAY
    )

    # Run simulation dequant
    recon_s, new_db_dequant_s = sim_binary_dequant_fastpath(
        packed_k, scale_k, base, delta_base, delta_decay_factor=DELTA_DECAY
    )
    
    # Compare dequant results
    assert_tensor_close(recon_k, recon_s, desc="Dequant: Reconstructed")
    assert_tensor_close(new_db_dequant_k, new_db_dequant_s, desc="Dequant: New Delta Base")


if __name__ == "__main__":
    pytest.main([__file__])
