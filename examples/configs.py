import torch
import torch.distributed as dist
from xfuser.compact.utils import CompactConfig, COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig

def get_config(model_name: str, method: str):
    if model_name == "flux":
        if method == "binary":
            config = _flux_binary_config()
        elif method == "lowrank12":
            config = _flux_lowrank12_config()
        elif method == "lowrank16":
            config = _flux_lowrank16_config()
        elif method == "df":
            config = _flux_distrifusion_config()
        elif method == "pipe":
            config = _flux_pipe_config()
        elif method == "ring":
            config = _flux_ring_config()
    elif model_name == "pixart":
        if method == "binary":
            config = _pixart_binary_config()
        elif method == "lowrank12":
            config = _pixart_lowrank12_config()
        elif method == "lowrank16":
            config = _pixart_lowrank16_config()
        elif method == "df":
            config = _pixart_distrifusion_config()
        elif method == "pipe":
            config = _pixart_pipe_config()
        elif method == "ring":
            config = _pixart_ring_config()
    else:
        raise ValueError(f"Model {model_name} not supported")
    assert isinstance(config, CompactConfig)
    return config

def _flux_binary_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.BINARY if step >= 2 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _flux_lowrank12_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=12,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _flux_lowrank16_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 2 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=16,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _flux_distrifusion_config():
    patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=2,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=None,
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _flux_pipe_config():
    return CompactConfig(
        enabled=False,
        compress_func=None,
        simulate=False,
        log_stats=False,
    )
    
def _flux_ring_config():
    return CompactConfig(
        enabled=False,
        compress_func=None,
        simulate=False,
        log_stats=False,
    )


def _pixart_binary_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.BINARY if step >= 4 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _pixart_lowrank12_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 4 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=12,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _pixart_lowrank16_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 4 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=16,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _pixart_distrifusion_config():
    patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=2,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=None,
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _pixart_pipe_config():
    return CompactConfig(
        enabled=False,
        compress_func=None,
        simulate=False,
        log_stats=False,
    )
    
def _pixart_ring_config():
    return CompactConfig(
        enabled=False,
        compress_func=None,
        simulate=False,
        log_stats=False,
    )
