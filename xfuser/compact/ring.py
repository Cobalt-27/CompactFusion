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

from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
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
    
from xfuser.compact.slowpath import set_current_lowrank_scale

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
AWL = True # attn aware lowrank
AWL_RANDOM_PERM = True # testing random scaling

def compact_update_awl_scale(q, k):
    # (bs, seq_len, head_cnt, head_size)
    """
    Calculates key token importance by sampling queries and computing attention scores.

    Args:
        q: Query tensor (bs, seq_len, head_cnt, head_size)
        k: Key tensor (bs, seq_len, head_cnt, head_size)
    """
    with torch.no_grad(): # No need to track gradients for importance calculation
        bs, seq_len, head_cnt, head_size = q.shape
        # 1. Sample Queries (e.g., 10%)
        sample_ratio = 0.05
        sample_size = max(1, int(seq_len * sample_ratio)) # Ensure at least one sample
        # Use torch.randperm for efficient random sampling without replacement
        sampled_indices = torch.randperm(seq_len, device=q.device)[:sample_size]
        sampled_q = q[:, sampled_indices, :, :] # Shape: (bs, sample_size, head_cnt, head_size)
        # 2. Compute Attention Scores (Q * K^T)
        # Permute Q: (bs, head_cnt, sample_size, head_size)
        sampled_q_permuted = sampled_q.permute(0, 2, 1, 3)
        # Permute K^T: (bs, head_cnt, head_size, seq_len)
        k_permuted_transposed = k.permute(0, 2, 3, 1) 
        # Calculate scores: (bs, head_cnt, sample_size, seq_len)
        # Using float32 for stability in score calculation
        attn_scores = torch.matmul(sampled_q_permuted.float(), k_permuted_transposed.float())
        # 3. Aggregate Scores
        # Sum across heads: (bs, sample_size, seq_len)
        scores_summed_heads = attn_scores.sum(dim=1)
        # Sum across sampled queries: (bs, seq_len) -> Importance per key token per batch item
        key_token_importance = scores_summed_heads.sum(dim=1).flatten()
        # Normalize scores (optional, depends on how importance is used)
        
        assert key_token_importance.shape == (bs * seq_len,), f"{key_token_importance.shape} != {(bs * seq_len,)}, bs: {bs}, seq_len: {seq_len}"
        top_k = int(bs * seq_len * 0.1)
        threshold = torch.topk(key_token_importance, top_k, largest=True).values[-1]
        # Create a new tensor with scale values (10 for top 10%, 1 for others)
        token_scale = torch.ones_like(key_token_importance)
        token_scale[key_token_importance >= threshold] = 10.0
        # lets perform a random perm here to test a random scale
        if AWL_RANDOM_PERM:
            token_scale = token_scale[torch.randperm(bs * seq_len)]
        set_current_lowrank_scale(token_scale)

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
    # (bs, seq_len, head_cnt, head_size)
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
    if AWL: 
        compact_update_awl_scale(q, k)
    out = None
    lse = None

    compress_type = compact_config().compress_func(mod_idx, current_iter)
    assert compact_config().error_feedback, "error feedback must be enabled"
    
    k_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-k"
    v_my_cache_key = f"{mod_idx}-{comm.rank%comm.world_size}-v"
    original_k_shape = k.shape 
    original_v_shape = v.shape
    k_compress_rank = compact_config().comp_rank
    v_compress_rank = compact_config().comp_rank
    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        # low rank as scale is expensive, we use lowrank with stride=2 for v
        k_compress_rank = -1 # k is not low-rank, so we use mean as scale
        if current_iter % 2 == 0:
            v_compress_rank = -1
        else:
            v_compress_rank = compact_config().comp_rank
    k_to_send = compact_compress(k_my_cache_key, k, compress_type, update_cache=True, override_rank=k_compress_rank)
    v_to_send = compact_compress(v_my_cache_key, v, compress_type, update_cache=True, override_rank=v_compress_rank)
    
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
                k_recv_cache_key, k_to_send, prev_compress_type, original_k_shape, update_cache=True, override_rank=k_compress_rank
            )
            v = compact_decompress(
                v_recv_cache_key, v_to_send, prev_compress_type, original_v_shape, update_cache=True, override_rank=v_compress_rank
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
        # (bs, seq_len, head_cnt, head_size)
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
    
    
    # if AWL:
    #     token_importance = lse.squeeze(dim=-1) # -> nhead, seq_len
    #     assert token_importance.shape == (q.shape[0], q.shape[1], k.shape[2]), f"{token_importance.shape} != {(q.shape[0], q.shape[1], k.shape[2])}, q.shape: {q.shape}, k.shape: {k.shape}"
    #     # token_importance = torch.exp(token_importance)
    #     token_importance = torch.sum(token_importance.float(), dim=(0,2)).flatten()
    #     # Set top 10% tokens to a scale of 10, others to a scale of 1
    #     num_tokens = token_importance.size(0)
    #     top_k = int(num_tokens * 0.05)  # Calculate how many tokens are in the top 10%
    #     # Find the threshold value for the top 10%
    #     threshold = torch.topk(token_importance, top_k, largest=True).values[-1]
    #     # Create a new tensor with scale values (10 for top 10%, 1 for others)
    #     token_scale = torch.ones_like(token_importance)
    #     token_scale[token_importance >= threshold] = 20.0
    #     # Replace token_importance with the scaled version
    #     token_importance = token_scale
    #     # print(f'token_importance 1%: {token_importance.quantile(0.01):.4f}, 99%: {token_importance.quantile(0.99):.4f}')
    #     # # token_importance = token_importance.pow(2)
    #     # token_importance = torch.softmax(token_importance, dim=-1)
    #     # print(f"token_importance 1%: {token_importance.quantile(0.01):.4f}, 99%: {token_importance.quantile(0.99):.4f}")
    #     assert token_importance.shape == (q.shape[1],), f"{token_importance.shape} != {(q.shape[1],)}, q.shape: {q.shape}"
    #     set_current_lowrank_scale(token_importance)
    
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    if compact_config().check_cache_consistency:
        compact_cache().check_consistency(group=process_group)
    return out, lse, None