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
        low_rank_dim=config.low_rank_dim
    )
    global _step
    _step = None

def compact_hello():
    if dist.get_rank() == 0:
        print(f"🐳  Compact initialized")
        print(f"🟦  Compact enabled" if _config.enable_compress else "🟫  Compact disabled")
        if _config.enable_compress:
            print(f"🟦  Fastpath" if _config.fastpath else "🟫  No fastpath")
            print(f"🟦  Simulate compress" if _config.simulate_compress else "🟫  No simulate compress")
            print(f"🟦  Check Consistency" if _config.check_cache_consistency else "🟫  No check consistency")
            print(f"🟦  Dump Activations" if _config.dump_activations else "🟫  No dump activations")
            print(f"🟦  Calculate Total Error" if _config.calc_total_error else "🟫  No calculate total error")

def compact_config():
    return _config

def compact_set_step(step):
    global _step
    _step = step

def compact_get_step():
    global _step
    return _step


def compact_cache():
    return _cache


def compact_reset():
    global _cache
    _cache = CompactCache(
        quantize=_config.quantized_cache, 
        low_rank_dim=_config.low_rank_dim
    )
    from xfuser.compact.stats import stats_clear
    stats_clear()
    global _step
    _step = None


@Profiler.prof_func("compact._compress_fn")
def _compress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE):
    if _config.simulate_compress:
        # NOTE: if simulation enabled, directly return the simulated compress-then-decompress result
        return sim_compress(x, compress_type, _config.sparse_ratio, _config.comp_rank)

    return slowpath_compress(x, compress_type, rank=_config.comp_rank, sparse_ratio=_config.sparse_ratio)

            
@Profiler.prof_func("compact._decompress_fn")
def _decompress_fn(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, shape: tuple):
    if _config.simulate_compress:
        return x.view(shape)  # no need for further decompression
    return slowpath_decompress(x, shape, compress_type, rank=_config.comp_rank, sparse_ratio=_config.sparse_ratio)

@Profiler.prof_func("compact.compact_compress")
def compact_compress(
    cache_key,
    x: torch.Tensor,
    compress_type: COMPACT_COMPRESS_TYPE,
    update_cache: bool = False,
):
    assert x.is_contiguous()
    assert _config.enable_compress
    original_shape = x.shape
    if len(x.shape) >= 4:
        x = x.view(-1, x.shape[-2] * x.shape[-1])
    elif len(x.shape) == 3:
        shape = (x.shape[0] * x.shape[1], x.shape[2])
        x = x.view(shape)
    assert x.ndim == 2
    # NOTE: reshaped to (N-token, channel)

    def cond_cache_put(key, val, delta):
        if update_cache:
            _cache.put(key, val, delta)

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        if _config.fastpath:
            x_t = x.transpose(0, 1)
            base = _cache.get_base(cache_key)
            if base is None:
                cond_cache_put(cache_key, x_t, None)
            else:
                cond_cache_put(cache_key, x_t, x_t - base)
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
            assert _config.compress_residual == 2 and compress_type == COMPACT_COMPRESS_TYPE.BINARY
            from xfuser.compact.fastpath import binary_quant_fastpath
            base = _cache.get_base(cache_key)
            delta_base = _cache.get_delta_base(cache_key)
            x_t = x.transpose(0, 1)
            q, scale_u, scale_v, new_base, new_delta_base = binary_quant_fastpath(
                x_t, base, delta_base, update_cache, delta_decay_factor=_config.delta_decay_factor
            )
            if update_cache:
                _cache.put(cache_key, new_base, new_delta_base)
            compressed = torch.cat([
                q.flatten().view(torch.half),
                scale_u.flatten(),
                scale_v.flatten()
            ])
            if _config.log_compress_stats:
                stats_log().log(
                    cache_key, 
                    base.transpose(0, 1), 
                    delta_base.transpose(0, 1), 
                    x, # before_comp_activation
                    new_base.transpose(0, 1), # recv_activation (reconstructed base is logged here for fastpath)
                    compressed, 
                    _config.compress_residual,
                    ref_activation_path=_config.ref_activation_path,
                    dump_activations=_config.dump_activations,
                    calc_total_error=_config.calc_total_error
                )
        else:
            if _config.compress_residual == 0:
                compressed = _compress_fn(x, compress_type)
                if _config.log_compress_stats:
                    reconstructed_local = _decompress_fn(compressed, compress_type, x.shape)
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
                compressed = _compress_fn(delta, compress_type)
                recv_delta = _decompress_fn(compressed, compress_type, x.shape)
                reconstructed = base + recv_delta
                cond_cache_put(cache_key, reconstructed, None)
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
                compressed = _compress_fn(delta_delta, compress_type)
                recv_delta_delta = _decompress_fn(compressed, compress_type, x.shape)
                new_base = base + delta_base + recv_delta_delta
                new_delta_base = delta_base + recv_delta_delta
                cond_cache_put(
                    cache_key,
                    new_base,
                    _decay_delta_base(new_delta_base),
                )
                if _config.log_compress_stats:
                    # Pass config paths/flags to log function
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
    assert _config.enable_compress
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

    if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
        val = compressed.view(shape)
        if _config.fastpath:
            base = _cache.get_base(cache_key)
            if base is None:
                cond_cache_put(cache_key, val.transpose(0, 1), None)
            else:
                cond_cache_put(cache_key, val.transpose(0, 1), val.transpose(0, 1) - base)
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
            assert _config.compress_residual == 2 and compress_type == COMPACT_COMPRESS_TYPE.BINARY
            from xfuser.compact.fastpath import binary_dequant_fastpath
            numel = torch.prod(torch.tensor(shape, dtype=torch.int64)).item()
            channel_size = shape[-1]
            u_size = channel_size
            v_size = shape[0]
            assert compressed.numel() == numel // 16 + u_size + v_size, f"Mismatch in compressed tensor size: expected {numel // 16 + u_size + v_size}, got {compressed.numel()}"
            q = compressed[:numel // 16]
            scale_u = compressed[numel // 16:numel // 16 + u_size].flatten().contiguous()
            scale_v = compressed[numel // 16 + u_size: numel // 16 + u_size + v_size].flatten().contiguous()
            q = q.view(torch.uint8).view(channel_size, -1).contiguous()

            reconstructed, new_delta_base = binary_dequant_fastpath(
                q,
                scale_u,
                scale_v,
                _cache.get_base(cache_key),
                _cache.get_delta_base(cache_key),
                delta_decay_factor=_config.delta_decay_factor,
            )
            if update_cache:
                _cache.put(cache_key, reconstructed, new_delta_base)
            reconstructed = reconstructed.transpose(0, 1)
        else:
            if _config.compress_residual == 0:
                reconstructed = _decompress_fn(compressed, compress_type, shape)
            elif _config.compress_residual == 1:
                base = _cache.get_base(cache_key)
                recv_delta = _decompress_fn(compressed, compress_type, shape)
                reconstructed = base + recv_delta
                cond_cache_put(cache_key, reconstructed, None)
            elif _config.compress_residual == 2:
                base = _cache.get_base(cache_key)
                delta_base = _cache.get_delta_base(cache_key)
                recv_delta_delta = _decompress_fn(compressed, compress_type, shape)
                reconstructed = base + delta_base + recv_delta_delta
                new_delta_base = delta_base + recv_delta_delta
                cond_cache_put(cache_key, reconstructed, _decay_delta_base(new_delta_base))
            else:
                raise ValueError("Invalid compress_residual value")
        return reconstructed.view(original_shape)
    raise RuntimeError("should not reach here")

