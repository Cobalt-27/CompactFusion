import torch
import torch.distributed as dist
from xfuser.prof import Profiler
from enum import Enum
from xfuser.compact.patchpara.df_utils import PatchConfig
import os

ALLOW_DEPRECATED = os.environ.get("COMPACT_ALLOW_DEPRECATED", "0") == "1"

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
    LOW_RANK = "low-rank"


class CompactConfig:

    def __init__(
        self,
        enabled: bool = False,
        override_with_patch_gather_fwd: bool = False,
        patch_gather_fwd_config: PatchConfig = None,
        compress_func: callable = None,
        sparse_ratio=None,
        comp_rank=None,
        residual: int = 0,
        ef: bool = False,
        simulate: bool = False,
        log_stats: bool = False,
        check_consist: bool = False,
        fastpath: bool = False,
        quantized_cache: bool = False,
        cache_low_rank_dim: int | None = None,
        ref_activation_path: str | None = None,
        dump_activations: bool = False,
        calc_total_error: bool = False,
        delta_decay_factor: float | None = None
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
            quantized_cache (bool): Enable quantization for base tensor in CompactCache.
            low_rank_dim (int | None): Dimension for low-rank cache compression of delta_base. None or 0 disables it.
            ref_activation_path (str | None): Path for dumping/loading reference activations.
            dump_activations (bool): If True and path is set, dump activations.
            calc_total_error (bool): If True and path is set, calculate error against reference.
            delta_decay_factor (float): Decay factor applied to delta_base in 2nd order residual.
        """
        self.enabled = enabled
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
        # Cache behavior flags
        self.quantized_cache = quantized_cache
        self.cache_low_rank_dim = cache_low_rank_dim
        # Updated attributes
        self.ref_activation_path = ref_activation_path
        self.dump_activations = dump_activations
        self.calc_total_error = calc_total_error
        self.delta_decay_factor = delta_decay_factor
        
        self.override_with_patch_gather_fwd = override_with_patch_gather_fwd
        self.patch_gather_fwd_config = patch_gather_fwd_config
        
        assert cache_low_rank_dim is None, "deprecated"
        assert self.quantized_cache is False, "deprecated"
        # assert self.compress_residual != 2, "deprecated"
        
        # Add assertion to prevent simultaneous dump and calc
        assert not (self.dump_activations and self.calc_total_error), \
            "Cannot dump activations and calculate total error in the same run. Set one to False."

        if residual == 2:
            assert ef, "2nd order compression requires error feedback enabled."
        if self.fastpath:
            assert ef, "Fastpath requires error feedback enabled."
            assert not simulate, "Fastpath does not support simulation."
        
        if self.override_with_patch_gather_fwd:
            assert self.patch_gather_fwd_config is not None, "patch_gather_fwd_config must be set if override_with_patch_gather_fwd is True"
        else:
            assert self.patch_gather_fwd_config is None, "patch_gather_fwd_config must be None if override_with_patch_gather_fwd is False"


from xfuser.compact.compress_quantize import quantize_int8, dequantize_int8
from xfuser.compact.compress_lowrank import subspace_iter


class CompactCache:
    def __init__(self, quantize=False):
        self.quantize = quantize
        self.base = {}
        self.delta_base = {}
        if quantize:
            assert ALLOW_DEPRECATED
        self.passed_count = 0

    # @Profiler.prof_func("compact.CompactCache.put")
    def put(self, key, base, delta_base):
        # Quantize base if needed
        if self.quantize:
            base = quantize_int8(base)
        self.base[key] = base

        # Compress or store delta_base
        if delta_base is not None:
            self.delta_base[key] = delta_base
        else:
            self.delta_base[key] = None

    # @Profiler.prof_func("compact.CompactCache.get_base")
    def get_base(self, key):
        base = self.base.get(key, None)
        if self.quantize:
            if base is not None:
                base = dequantize_int8(*base)
        return base

    # @Profiler.prof_func("compact.CompactCache.get_delta_base") 
    def get_delta_base(self, key):
        # Retrieve stored item for delta_base
        stored_item = self.delta_base.get(key, None)
        return stored_item

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
            # Reconstruct/retrieve delta_base before checking
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