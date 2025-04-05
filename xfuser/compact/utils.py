import torch
import torch.distributed as dist
from xfuser.prof import Profiler
from enum import Enum


class COMPACT_COMPRESS_TYPE(Enum):
    """
    Enumeration of compression types for compact communication.

    SPARSE: Uses top-k sparsity to compress tensors
    QUANT: Uses quantization to compress tensors
    HYBRID: Combines topk sparsity and quantization for compression
    """

    WARMUP = "warmup"
    SPARSE = "sparse"
    BINARY = "binary"
    INT2 = "int2"
    IDENTITY = "identity"  # go thorugh the entire pipeline, but no compression
    BINARY_SPARSE = "binary-sparse"
    INT2_SPARSE = "int2-sparse"
    BINARY_LOW_RANK = "binary-low-rank"
    INT2_LOW_RANK = "int2-low-rank"
    LOW_RANK = "low-rank"


class CompactConfig:

    def __init__(
        self,
        enabled: bool = False,
        compress_func: callable = None,
        sparse_ratio=None,
        comp_rank=None,
        residual: int = 0,
        ef: bool = False,
        simulate: bool = False,
        log_stats: bool = False,
        check_consist: bool = False,
        fastpath: bool = False,
    ):
        """
        Initialize compression settings.
        Args:
            enabled (bool): Enable/disable compression.
            compress_func (callable): (layer_idx, step) -> compress_type, step starts from 0.
            residual (int): 0: no residual, 1: 1st order residual, 2: 2nd order residual.
            ef (bool): Enable/disable EF compression.
            simulate (bool): Enable/disable simulation compression.
            log_stats (bool): Enable/disable logging of compression stats.
        """
        self.enable_compress = enabled
        self.compress_func = compress_func
        self.sparse_ratio = sparse_ratio
        self.comp_rank = comp_rank
        assert residual in [0, 1, 2]
        self.compress_residual = residual
        self.error_feedback = ef
        self.simulate_compress = simulate
        # STATS related
        self.log_compress_stats = log_stats
        self.check_cache_consistency = check_consist
        self.fastpath = fastpath
        
        if residual == 2:
            assert ef, "2nd order compression requires error feedback enabled."
        if self.fastpath:
            assert residual == 2, "Fastpath requires 2nd order compression."
            assert ef, "Fastpath requires error feedback enabled."
            assert not simulate, "Fastpath does not support simulation."


from xfuser.compact.compress_quantize import quantize_int8, dequantize_int8


class CompactCache:
    def __init__(self, quantize=False):
        self.quantize = quantize
        self.base = {}
        self.delta_base = {}
        self.passed_count = 0

    def put(self, key, base, delta_base):
        if delta_base is not None:
            assert base.shape == delta_base.shape
        if self.quantize:
            base = quantize_int8(base)
            delta_base = quantize_int8(delta_base) if delta_base is not None else None
        self.base[key] = base
        self.delta_base[key] = delta_base

    def get_base(self, key):
        base = self.base.get(key, None)
        if self.quantize:
            if base is not None:
                base = dequantize_int8(*base)
        return base

    def get_delta_base(self, key):
        delta_base = self.delta_base.get(key, None)
        if self.quantize:
            if delta_base is not None:
                delta_base = dequantize_int8(*delta_base)
        return delta_base

    def check_consistency(self, group=None):
        """
        Checks cache consistency for all keys across all GPUs in the specified group.
        Args:
            group: Optional process group to check consistency within. If None, uses the default world group.
        """
        if group is None:
            group = dist.group.WORLD
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return # No need for consistency check with a single process
        # Iterate through all keys present in the local cache
        # Assumes all ranks have the same keys
        for key in sorted(self.base.keys()):
            local_base = self.get_base(key)
            local_delta_base = self.get_delta_base(key)

            # Flatten and concatenate tensors if they exist
            tensors_to_check = []
            if local_base is not None:
                tensors_to_check.append(local_base.flatten())
            if local_delta_base is not None:
                tensors_to_check.append(local_delta_base.flatten())
            
            if tensors_to_check:
                # Concatenate all tensors into a single flat tensor
                combined_tensor = torch.cat(tensors_to_check)
                tensor_to_reduce = combined_tensor.clone().detach().float()
                dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM, group=group)
                average_tensor = tensor_to_reduce / world_size
                assert torch.allclose(combined_tensor.float(), average_tensor, atol=1e-3), f'Inconsistent cache at key {key}, max diff: {torch.max(torch.abs(combined_tensor.float() - average_tensor)):.6f}'
        self.passed_count += 1

class PowerCache:
    """
    Cache for supspace iteration
    Not used for now
    """
    def __init__(self):
        self.cache = {}

    def put(self, key, value):
        if key in self.cache:
            assert value.shape == self.cache[key].shape
        self.cache[key] = value
    
    def get(self, key):
        return self.cache.get(key, None)


def get_emoji():
    import random
    emojis = [
        "☝️ 😅",
        "👊🤖🔥",
        "🙏 🙏 🙏",
        "🐳🌊🐚",
        "☘️ ☘️ 🍀",
        "🎊🎉🎆",
        "🌇🌆🌃",
        "🍾🍾🍾",
        "🅰  🅲  🅲  🅴  🅿  🆃  🅴  🅳",
        "🖼️ 🖌️ 🎨",
        "🐳  🅲  🅾  🅼  🅿  🅰  🅲  🆃",
        "╰(*°▽°*)╯",
        "ヾ(≧▽≦*)o",
        "⚡️ 🔗 ⚡️",
        "💾 ➡️ 🚀"
    ]
    return random.choice(emojis)