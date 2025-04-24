import torch
import torch.distributed as dist
from xfuser.compact.utils import (
    CompactConfig,
    CompactCache,
    COMPACT_COMPRESS_TYPE,
)
from xfuser.prof import Profiler
# from xfuser.modules.base_module import BaseModule
from xfuser.compact.stats import stats_log
import os
from xfuser.compact.slowpath import slowpath_compress, slowpath_decompress, sim_compress
from xfuser.compact.patchpara.df_cache import AllGatherCache
from xfuser.compact.utils import ALLOW_DEPRECATED

"""
COMPACT: Activation Compression with Delta Transmission and Error Feedback

In diffusion models, activations change only slightly from one denoising step to the next.
Simply transmitting the full activations incures high redundancy.
Rather than transmitting full activations between nodes, 
we transmit only the difference (delta) between the current activation and a maintained baseline. 
This delta is highly compressible.

Our approach works as follows:
1. Compute the delta as the difference between the current activation and a cached baseline.
2. Compress and transmit the delta.
3. At the receiver, decompress the delta and add it to the baseline to reconstruct the activation.
4. Both sender and receiver update the baseline with the same reconstructed activation, so that any error is compensated in future.

This error feedback mechanism ensures that, over time, the compression errors do not accumulate,
leading to accurate reconstruction while significantly reducing communication overhead.
"""


def compact_init(config: CompactConfig):
    global _config
    _config = config
    global _cache
    # Initialize cache using flags from the provided config
    _cache = CompactCache(
        quantize=config.quantized_cache,
    )
    global _step
    _step = None
    if config.override_with_patch_gather_fwd:
        global _allgather_cache
        _allgather_cache = AllGatherCache()
    
    global _current_cache_key
    _current_cache_key = None

def compact_hello():
    if dist.get_rank() == 0:
        print(f"🐳  Compact initialized")
        print(f"🟦  Compact enabled" if _config.enabled else "🟫  Compact disabled")
        if _config.enabled:
            if not _config.override_with_patch_gather_fwd:
                print(f"🟦  Fastpath" if _config.fastpath else "🟫  No fastpath")
                print(f"🟦  Simulate compress" if _config.simulate_compress else "🟫  No simulate compress")
                print(f"🟦  Stats log" if _config.log_compress_stats else "🟫  No stats log")
                print(f"🟦  Check consistency" if _config.check_cache_consistency else "🟫  No check consistency")
                print(f"🟦  Dump activations" if _config.dump_activations else "🟫  No dump activations")
                print(f"🟦  Calculate total error" if _config.calc_total_error else "🟫  No calculate total error")
            else:
                print(f"🟧  Overrided to Patch Para")
                patch_config = _config.patch_gather_fwd_config
                print(f"🟨  Using DistriFusion" if patch_config.async_comm else "🟫  Sync patch para")

def compact_config():
    global _config
    return _config

def compact_set_step(step):
    global _step
    _step = step

def compact_get_step():
    global _step
    return _step


def compact_cache():
    return _cache

def allgather_cache():
    global _allgather_cache
    return _allgather_cache

def compact_reset():
    global _cache
    _cache = CompactCache(
        quantize=_config.quantized_cache, 
    )
    from xfuser.compact.stats import stats_clear
    stats_clear()
    global _step
    _step = None
    if _config.override_with_patch_gather_fwd:
        global _allgather_cache
        _allgather_cache = AllGatherCache()
    global _current_cache_key
    _current_cache_key = None

def compact_get_current_cache_key():
    """
    FOR TESTING ONLY
    """
    global _current_cache_key
    return _current_cache_key

@Profiler.prof_func("compact._compress_fn")
def _compress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, rank: int):
    if _config.simulate_compress:
        # NOTE: if simulation enabled, directly return the simulated compress-then-decompress result
        return sim_compress(x, compress_type, _config.sparse_ratio, rank)

    return slowpath_compress(x, compress_type, rank=rank, sparse_ratio=_config.sparse_ratio)

            
@Profiler.prof_func("compact._decompress_fn")
def _decompress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, shape: tuple, rank: int):
    if _config.simulate_compress:
        return x.view(shape)  # no need for further decompression
    return slowpath_decompress(x, shape, compress_type, rank=rank, sparse_ratio=_config.sparse_ratio)

@Profiler.prof_func("compact.compact_compress")
def compact_compress(
    cache_key,
    x: torch.Tensor,
    compress_type: COMPACT_COMPRESS_TYPE,
    update_cache: bool = False,
):
    global _current_cache_key
    _current_cache_key = cache_key
    assert x.is_contiguous()
    assert _config.enabled
    original_shape = x.shape
    if len(x.shape) >= 4:
        x = x.view(-1, x.shape[-2] * x.shape[-1])
    elif len(x.shape) == 3:
        shape = (x.shape[0] * x.shape[1], x.shape[2])
        x = x.view(shape)
    assert x.ndim == 2
    # NOTE: reshaped to (N-token, channel)
    rank = _config.comp_rank
    if compress_type == COMPACT_COMPRESS_TYPE.BINARY and rank != -1:
        assert ALLOW_DEPRECATED, "Binary compression with rank != -1 is deprecated"
    
    def cond_cache_put(key, val, delta):
        if update_cache:
            _cache.put(key, val, delta)

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        if _config.fastpath:
            assert _config.compress_residual == 1
            # Cache directly in N, C format
            cond_cache_put(cache_key, x, None)
        else:
            if _config.compress_residual == 1:
                cond_cache_put(cache_key, x, None)
            elif _config.compress_residual == 2:
                base = _cache.get_base(cache_key)
                if base is None:
                    cond_cache_put(cache_key, x, None)
                else:
                    cond_cache_put(cache_key, x, x - base)
        return x.view(original_shape)
    else:
        if _config.fastpath:
            assert compress_type == COMPACT_COMPRESS_TYPE.BINARY
            assert _config.compress_residual == 1
            from xfuser.compact.fastpath import binary_quant_fastpath
            # Base should be (N, C) now
            base = _cache.get_base(cache_key) 
            # Call with N, C layout
            q, scale_u_ck, scale_v_kn, new_base = binary_quant_fastpath(
                x_tensor_nc=x,
                base_tensor_nc=base,
                rank=rank,
                update_cache=update_cache,
            )
            # q: (N, C//8), scale_u_nk: (N, K), scale_v_ck: (C, K), new_base: (N, C)
            if update_cache:
                # new_base is already N, C
                _cache.put(cache_key, new_base, None)

            q_flat_half = q.view(torch.half).flatten()
            u_flat_half = scale_u_ck.flatten() # New: U is (N, K)
            v_flat_half = scale_v_kn.flatten() # New: V is (C, K)
            compressed = torch.cat([q_flat_half, u_flat_half, v_flat_half], dim=0)

            if _config.log_compress_stats:
                log_base = base # Already N, C
                log_recv_activation = new_base # Already N, C
                stats_log().log(
                    cache_key,
                    log_base,
                    None,
                    x,
                    log_recv_activation,
                    compressed,
                    1,
                    ref_activation_path=_config.ref_activation_path,
                    dump_activations=_config.dump_activations,
                    calc_total_error=_config.calc_total_error
                )
        else:
            if _config.compress_residual == 0:
                compressed = _compress_fn(x, compress_type, rank)
                if _config.log_compress_stats:
                    reconstructed_local = _decompress_fn(compressed, compress_type, x.shape, rank)
                    stats_log().log(
                        cache_key, 
                        base=None, 
                        delta_base=None, 
                        before_comp_activation=x, 
                        recv_activation=reconstructed_local, 
                        compressed_tensor=compressed, 
                        compress_residual=_config.compress_residual,
                        ref_activation_path=_config.ref_activation_path,
                        dump_activations=_config.dump_activations,
                        calc_total_error=_config.calc_total_error
                    )
            elif _config.compress_residual == 1:
                base = _cache.get_base(cache_key)
                delta = x - base
                compressed = _compress_fn(delta, compress_type, rank)
                recv_delta = _decompress_fn(compressed, compress_type, x.shape, rank)
                reconstructed = base + recv_delta
                cond_cache_put(cache_key, reconstructed if _config.error_feedback else x, None)
                if _config.log_compress_stats:
                    stats_log().log(
                        cache_key, 
                        base=base, 
                        delta_base=None, 
                        before_comp_activation=x, 
                        recv_activation=reconstructed, 
                        compressed_tensor=compressed, 
                        compress_residual=_config.compress_residual,
                        ref_activation_path=_config.ref_activation_path,
                        dump_activations=_config.dump_activations,
                        calc_total_error=_config.calc_total_error
                    )
            elif _config.compress_residual == 2:
                base = _cache.get_base(cache_key)
                delta_base = _cache.get_delta_base(cache_key)
                delta_delta = x - base - delta_base
                compressed = _compress_fn(delta_delta, compress_type, rank)
                recv_delta_delta = _decompress_fn(compressed, compress_type, x.shape, rank)
                new_base = base + delta_base + recv_delta_delta
                new_delta_base = delta_base + recv_delta_delta
                cond_cache_put(
                    cache_key,
                    new_base,
                    _decay_delta_base(new_delta_base),
                )
                if _config.log_compress_stats:
                    stats_log().log(
                        cache_key, 
                        base, 
                        delta_base, 
                        x, # before_comp_activation
                        new_base, # recv_activation 
                        compressed, 
                        _config.compress_residual,
                        ref_activation_path=_config.ref_activation_path,
                        dump_activations=_config.dump_activations,
                        calc_total_error=_config.calc_total_error
                    )
            else:
                raise ValueError("Invalid compress_residual value")
        return compressed
    raise RuntimeError("should not reach here")

def _decay_delta_base(delta_base):
    return delta_base * _config.delta_decay_factor

@Profiler.prof_func("compact.compact_decompress")
def compact_decompress(
    cache_key,
    compressed: torch.Tensor,
    compress_type: COMPACT_COMPRESS_TYPE,
    shape: tuple,
    update_cache: bool = False,
):
    global _current_cache_key
    _current_cache_key = cache_key
    assert _config.enabled
    original_shape = shape
    if len(shape) >= 4:
        # TODO: check tensor layout for all_gather
        dim_0 = 1
        for i in range(len(shape) - 2):
            dim_0 *= shape[i]
        shape = (dim_0, shape[-2] * shape[-1])
    elif len(shape) == 3:
        shape = (shape[0] * shape[1], shape[2])
    else:
        assert len(shape) == 2

    def cond_cache_put(key, val, delta):
        if update_cache:
            _cache.put(key, val, delta)
    rank = _config.comp_rank
    if rank == -1:
        rank = 1 # effective rank is 1 for mean

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        val = compressed.view(shape)
        if _config.fastpath:
            assert _config.compress_residual == 1
            # Cache directly in N, C format
            cond_cache_put(cache_key, val, None)
        else:
            if _config.compress_residual == 1:
                cond_cache_put(cache_key, val, None)
            elif _config.compress_residual == 2:
                base = _cache.get_base(cache_key)
                if base is None:
                    cond_cache_put(cache_key, val, None)
                else:
                    cond_cache_put(cache_key, val, val - base)
        return val.view(original_shape)
    else:
        if _config.fastpath:
            assert compress_type == COMPACT_COMPRESS_TYPE.BINARY, "Fastpath only supports BINARY compress type"
            assert _config.compress_residual == 1
            from xfuser.compact.fastpath import binary_dequant_fastpath

            N, C = shape # Shape is N, C after initial reshape
            assert C % 8 == 0, "Channel dim C must be divisible by 8 for binary fastpath"

            # Calculate split sizes for packed(N,C//8 uint8), scale_u(N,K half), scale_v(C,K half)
            q_numel_uint8 = N * (C // 8)
            q_numel_half = q_numel_uint8 // 2 # Stored as FP16 view
            u_numel_half = N * rank             # U is (N, K)
            v_numel_half = C * rank             # V is (C, K)

            expected_numel = q_numel_half + u_numel_half + v_numel_half
            assert compressed.numel() == expected_numel, \
                f"Mismatch in compressed tensor size: expected {expected_numel} (q={q_numel_half}, u={u_numel_half}, v={v_numel_half}), got {compressed.numel()}, Shape (N,C)=({N},{C}), Rank={rank}"

            # Split the compressed tensor
            q_half, scale_u_flat, scale_v_flat = torch.split(
                compressed,
                [q_numel_half, u_numel_half, v_numel_half]
            )

            # Reshape
            packed_nc8 = q_half.view(torch.uint8).view(N, C // 8)
            scale_u_nk = scale_u_flat.view(N, rank)
            scale_v_ck = scale_v_flat.view(C, rank)

            base_nc = _cache.get_base(cache_key)

            # Updated function call signature and return value
            reconstructed_nc = binary_dequant_fastpath(
                packed_in_nc8=packed_nc8,
                scale_u_nk=scale_u_nk,
                scale_v_ck=scale_v_ck,
                base_nc=base_nc,
                # rank=rank,
            )
            # reconstructed_nc is (N, C)
            if update_cache:
                _cache.put(cache_key, reconstructed_nc, None)
            reconstructed = reconstructed_nc # Already (N, C)
        else:
            if _config.compress_residual == 0:
                reconstructed = _decompress_fn(compressed, compress_type, shape, rank)
            elif _config.compress_residual == 1:
                base = _cache.get_base(cache_key)
                recv_delta = _decompress_fn(compressed, compress_type, shape, rank)
                reconstructed = base + recv_delta
                cond_cache_put(cache_key, reconstructed, None)
            elif _config.compress_residual == 2:
                base = _cache.get_base(cache_key)
                delta_base = _cache.get_delta_base(cache_key)
                recv_delta_delta = _decompress_fn(compressed, compress_type, shape, rank)
                reconstructed = base + delta_base + recv_delta_delta
                new_delta_base = delta_base + recv_delta_delta
                cond_cache_put(cache_key, reconstructed, _decay_delta_base(new_delta_base))
            else:
                raise ValueError("Invalid compress_residual value")
        return reconstructed.view(original_shape)
    raise RuntimeError("should not reach here")

def compact_all_gather(
    tag,
    x: torch.Tensor,
    comp_type: COMPACT_COMPRESS_TYPE,
    group=None,
):
    # raise NotImplementedError("Compact all gather is inconsistent with ring impl.")
    assert _config.enabled
    rank = dist.get_rank(group)
    my_key = f"{tag}-{rank}"
    to_send = compact_compress(
        my_key,
        x,
        comp_type,
        update_cache=False,
    )
    world_size = dist.get_world_size(group)
    buf_list = [torch.empty_like(to_send) for _ in range(world_size)]
    with Profiler.scope("compact.all_gather"):
        dist.all_gather(buf_list, to_send, group=group, async_op=False)
    decompressed_list = [
        compact_decompress(
            f"{tag}-{i}",
            buf,
            comp_type,
            x.shape,
            update_cache=True,
        )
        for i, buf in enumerate(buf_list)
    ]
    return decompressed_list