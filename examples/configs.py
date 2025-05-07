import torch
import torch.distributed as dist
from xfuser.compact.utils import CompactConfig, COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig

def get_config(model_name: str, method: str):
    print(f"get_config: model_name={model_name}, method={method}")
    if model_name in ["Flux", "Pixart-alpha"]:
        if method == "binary":
            config = _binary_config()
        elif method == "int2":
            config = _int2_config()
        elif method == "lowrank12":
            config = _lowrank12_config()
        elif method == "lowrank8":
            config = _lowrank8_config()
        elif method == "lowrankq32":
            config = _lowrankq32_config()
        # elif method == "lowrank16":
        #     config = _flux_lowrank16_config()
        elif method == "df":
            config = _distrifusion_config()
        elif method == "pipe":
            config = _disabled_config()
        elif method == "ring":
            config = _disabled_config()
        elif method == "patch":
            config = _patch_config()
        elif method == "ulysses":
            config = _disabled_config()
        elif method == "int2patch":
            config = _int2_patch_config()
    else:
        raise ValueError(f"Model {model_name} not supported")
    assert isinstance(config, CompactConfig)
    return config

def _binary_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.BINARY if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )
    
def _int2_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.INT2 if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _lowrank12_config():
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

def _lowrank8_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=8,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrankq32_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK_Q if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=32,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrank16_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=16,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _int2_patch_config():
    patch_config = PatchConfig(
        use_compact=True,
        async_comm=False,
        async_warmup=1,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.INT2 if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _distrifusion_config():
    patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=1,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=None,
        ef=False,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _disabled_config():
    return CompactConfig(
        enabled=False,
        compress_func=None,
        simulate=False,
        log_stats=False,
    )

def _patch_config():
    patch_config = PatchConfig(
        use_compact=False,
        async_comm=False,
        async_warmup=1,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=None,
        ef=False,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrank_config():
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= 1 else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=8,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )
