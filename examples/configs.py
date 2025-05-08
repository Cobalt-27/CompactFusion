import torch
import torch.distributed as dist
from xfuser.compact.utils import CompactConfig, COMPACT_COMPRESS_TYPE
from xfuser.compact.patchpara.df_utils import PatchConfig

def get_config(model_name: str, method: str):
    print(f"get_config: model_name={model_name}, method={method}")
    if model_name in ["Flux", "Pixart-alpha", "CogVideoX"]:
        warmup_steps = 2 if model_name == "CogVideoX" else 1
        if method == "binary":
            config = _binary_config(warmup_steps)
        elif method == "int2":
            config = _int2_config(warmup_steps)
        elif method == "lowrank12":
            config = _lowrank12_config(warmup_steps)
        elif method == "lowrank8":
            config = _lowrank8_config(warmup_steps)
        elif method == "lowrankq32":
            config = _lowrankq32_config(warmup_steps)
        # elif method == "lowrank16":
        #     config = _flux_lowrank16_config()
        elif method == "df":
            config = _distrifusion_config(warmup_steps)
        elif method == "pipe":
            config = _disabled_config()
        elif method == "ring":
            config = _disabled_config()
        elif method == "patch":
            config = _patch_config(warmup_steps)
        elif method == "ulysses":
            config = _disabled_config()
        elif method == "int2patch":
            config = _int2_patch_config(warmup_steps)
    else:
        raise ValueError(f"Model {model_name} not supported")
    assert isinstance(config, CompactConfig)
    return config

def _binary_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.BINARY if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )
    
def _int2_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.INT2 if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _lowrank12_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=12,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrank8_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=8,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrankq32_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK_Q if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=32,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _lowrank16_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=16,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )

def _int2_patch_config(warmup_steps: int):
    patch_config = PatchConfig(
        use_compact=True,
        async_comm=False,
        async_warmup=warmup_steps,
    )
    return CompactConfig(
        enabled=True,
        override_with_patch_gather_fwd=True,
        patch_gather_fwd_config=patch_config,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.INT2 if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=-1,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=True,
    )

def _distrifusion_config(warmup_steps: int):
    patch_config = PatchConfig(
        use_compact=False,
        async_comm=True,
        async_warmup=warmup_steps,
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
        async_warmup=0,
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

def _lowrank_config(warmup_steps: int):
    return CompactConfig(
        enabled=True,
        compress_func=lambda layer_idx, step: COMPACT_COMPRESS_TYPE.LOW_RANK if step >= warmup_steps else COMPACT_COMPRESS_TYPE.WARMUP,
        comp_rank=8,
        residual=1, # 0 for no residual, 1 for delta, 2 for delta-delta
        ef=True,
        simulate=False,
        log_stats=False,
        fastpath=False,
    )
