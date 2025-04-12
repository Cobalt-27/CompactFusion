"""
NOTE: code from yunchang
"""

import torch
import torch.distributed as dist

from yunchang.ring.utils import RingComm, update_out_and_lse


from yunchang import LongContextAttention
from yunchang.kernels import AttnType
from yunchang.hybrid.utils import RING_IMPL_DICT
from yunchang.kernels import select_flash_attn_impl
from yunchang.comm.all_to_all import SeqAllToAll4D
from torch import Tensor

from xfuser.compact.utils import CompactCache, CompactCache
from xfuser.compact.main import (
    compact_config,
    compact_cache,
    compact_compress,
    compact_decompress,
)
from xfuser.prof import Profiler

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
except ImportError:
    flash_attn = None
    _flash_attn_forward = None
    from yunchang.kernels.attention import pytorch_attn_forward

def compact_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    return_attn_probs=None,
    deterministic=False,
    attn_layer=None,
    group=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    mod_idx=None,
    current_iter=None,
):
    """
    Compact ring attention forward pass.
    """
    gather = compact_config().override_with_patch_gather_fwd
    
    if gather:
        from xfuser.compact.patchpara.fwd import patch_gather_fwd
        return patch_gather_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter
        )
    else:
        return _compact_ring_fwd(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_attn_probs,
            deterministic, attn_layer, group, joint_tensor_key, joint_tensor_value, joint_strategy, mod_idx, current_iter
        )

@Profiler.prof_func("compact._compact_ring_fwd")
def _compact_ring_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    return_attn_probs=None,
    deterministic=False,
    attn_layer=None,
    group=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="none",
    mod_idx=None,
    current_iter=None,
):
    assert alibi_slopes is None
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    is_joint = False
    if (joint_tensor_key is not None and 
        joint_tensor_value is not None):
        supported_joint_strategy = ["front", "rear"]
        if joint_strategy not in supported_joint_strategy:
            raise ValueError(
                f"joint_strategy: {joint_strategy} not supprted. supported joint strategy: {supported_joint_strategy}"
            )
        else:
            is_joint = True
    elif (joint_tensor_key is None and 
        joint_tensor_value is None):
        pass
    else:
        raise ValueError(
            f"joint_tensor_key and joint_tensor_value should be None or not None simultaneously."
        )
    if attn_layer is not None:
        # XXX: we dont need KV Cache in ring
        # k, v = get_cache_manager().update_and_get_kv_cache(
        #     new_kv=[k, v],
        #     layer=attn_layer,
        #     slice_dim=1,
        #     layer_type="attn",
        # )
        # k = k.contiguous()
        # v = v.contiguous()
        pass
    process_group = group
    comm = RingComm(process_group)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = None
    lse = None

    compress_type = compact_config().compress_func(mod_idx, current_iter)
    assert compact_config().error_feedback, "error feedback must be enabled"
    
    k_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-k"
    v_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-v"
    original_k_shape = k.shape 
    original_v_shape = v.shape
    k_to_send = compact_compress(k_my_cache_key, k, compress_type, update_cache=True)
    v_to_send = compact_compress(v_my_cache_key, v, compress_type, update_cache=True)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            buf_k: torch.Tensor = comm.send_recv(k_to_send)
            buf_v: torch.Tensor = comm.send_recv(v_to_send)
            comm.commit()
        
        if step != 0:
            recv_rank = (comm.rank - step) % comm.world_size
            k_recv_cache_key = f"{mod_idx}-{recv_rank}-k"
            v_recv_cache_key = f"{mod_idx}-{recv_rank}-v"
            k = compact_decompress(
                k_recv_cache_key, k_to_send, prev_compress_type, original_k_shape, update_cache=True
            )
            v = compact_decompress(
                v_recv_cache_key, v_to_send, prev_compress_type, original_v_shape, update_cache=True
            )
        k = k.contiguous() 
        v = v.contiguous()

        if is_joint and joint_strategy == "rear":
            if step + 1 == comm.world_size:
                key_to_use = torch.cat([k, joint_tensor_key], dim=1)
                value_to_use = torch.cat([v, joint_tensor_value], dim=1)
            else:
                key_to_use, value_to_use = k, v
        elif is_joint and joint_strategy == "front":
            if step == 0:
                key_to_use = torch.cat([joint_tensor_key, k], dim=1)
                value_to_use = torch.cat([joint_tensor_value, v], dim=1)
            else:
                key_to_use, value_to_use = k, v
        else:
            key_to_use, value_to_use = k, v

        if not causal or step <= comm.rank:
            if flash_attn is None:
                block_out, block_lse = pytorch_attn_forward(
                    q,
                    key_to_use,
                    value_to_use,
                    dropout_p,
                    softmax_scale,
                    causal=causal and step == 0,
                )
            else:
                if flash_attn.__version__ <= "2.6.3": 
                    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                        q,
                        key_to_use,
                        value_to_use,
                        dropout_p,
                        softmax_scale,
                        causal=causal and step == 0,
                        window_size=window_size,
                        softcap=0.0,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True and dropout_p > 0,
                    )
                else:
                     block_out, block_lse, _, _ = _flash_attn_forward(
                        q,
                        key_to_use,
                        value_to_use,
                        dropout_p,
                        softmax_scale,
                        causal=causal and step == 0,
                        window_size_left=window_size[0],
                        window_size_right=window_size[1],
                        softcap=0.0,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True and dropout_p > 0,
                    )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            with Profiler.scope("compact.ring.wait"):
                comm.wait()
            k_to_send = buf_k 
            v_to_send = buf_v
            prev_compress_type = compress_type

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    if compact_config().check_cache_consistency:
        compact_cache().check_consistency(group=process_group)
    return out, lse, None