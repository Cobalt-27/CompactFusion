"""
NOTE: code from yunchang
"""

import torch
import torch.distributed as dist

# from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from yunchang.ring.utils import RingComm, update_out_and_lse


from yunchang.hybrid.utils import RING_IMPL_DICT
from yunchang.kernels import select_flash_attn_impl, FlashAttentionImpl
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP
from torch import Tensor

from pipefuser.compact.utils import CompactCache, CompactCache
from pipefuser.compact.main import (
    compact_config,
    compact_cache,
    compact_compress,
    compact_decompress,
)
from pipefuser.prof import Profiler


class CompactSPAttn(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_sync: bool = False,
        attn_type: FlashAttentionImpl = FlashAttentionImpl.FA,
    ) -> None:

        super(CompactSPAttn, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        # assert dist.get_world_size(self.ulysses_pg) == 1, "Ulysses not supported"

        self.use_pack_qkv = use_pack_qkv
        self.use_sync = use_sync
        self.attn_type = attn_type
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        # self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]
        # NOTE: using our own ring implementation

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        mod_idx=None,
        mod_counter=None,
        *args,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg,
                qkv,
                self.scatter_idx,
                self.gather_idx,
                use_sync=self.use_sync,
            )
            qkv = torch.chunk(qkv, 3, dim=0)
            out = _compact_ring_fwd(
                qkv[0],
                qkv[1],
                qkv[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                mod_idx=mod_idx,
                mod_counter=mod_counter,
            )
        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx, self.use_sync
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx, self.use_sync
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx, self.use_sync
            )

            out = _compact_ring_fwd(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                mod_idx=mod_idx,
                mod_counter=mod_counter,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg,
            context_layer,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )

        # out e.g., [s/p::h]
        return output

    def backward(self, dout, *args):
        pass

@Profiler.prof_func("compact._compact_ring_fwd")
def _compact_ring_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    group=None,
    attn_type: FlashAttentionImpl = FlashAttentionImpl.FA,
    return_attn_probs=False,
    mod_idx=None,
    mod_counter=None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    assert alibi_slopes is None
    process_group = group
    comm = RingComm(process_group)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = None
    lse = None

    compress_type = compact_config().compress_func(mod_idx, mod_counter)
    assert compact_config().error_feedback, "error feedback must be enabled"
    
    # Stack k and v along the first dimension (dim=0)
    # Shape: (2, bs, seq_len, num_heads, head_size)
    kv = torch.stack([k, v], dim=0).contiguous() # Keep contiguous for the stacked tensor

    # Use a single cache key for the combined kv tensor
    kv_my_cache_key = f"{mod_idx}-{comm.rank % comm.world_size}-kv"
    kv_to_send = compact_compress(kv_my_cache_key, kv, compress_type, update_cache=True)

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            # Send/receive the combined tensor
            buf_kv: torch.Tensor = comm.send_recv(kv_to_send)
            comm.commit()
        
        if step != 0:
            recv_rank = (comm.rank - step) % comm.world_size
            # Use a single cache key for received kv
            kv_recv_cache_key = f"{mod_idx}-{recv_rank}-kv"
            # Decompress the combined tensor
            # Need the original shape of the *combined* tensor for decompression
            kv = compact_decompress(
                kv_recv_cache_key, kv_to_send, prev_compress_type, kv.shape, update_cache=True
            )
        # else: kv is the original local kv from the first step

        # Split the combined tensor back into k and v using slicing (zero-copy)
        # k_split shape: (bs, seq_len, num_heads, head_size)
        # v_split shape: (bs, seq_len, num_heads, head_size)
        k_split = kv[0].contiguous() # Use .contiguous() just in case attn kernel requires it
        v_split = kv[1].contiguous() # Use .contiguous() just in case attn kernel requires it

        if not causal or step <= comm.rank:
            fn = select_flash_attn_impl(attn_type, stage="fwd-only")
            block_out, block_lse = fn(
                q,
                k_split, # Use the split k
                v_split, # Use the split v
                dropout_p,
                softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            with Profiler.scope("compact.ring.wait"):
                comm.wait()
            # Update the tensor to send for the next step
            kv_to_send = buf_kv
            # Keep track of the compression type used for the *received* tensor
            # which becomes the *previous* type for the *next* decompression
            prev_compress_type = compress_type # Assuming compress_type doesn't change mid-ring, otherwise need to sync this too

        # _check_cache_consistency(mod_idx, comm)

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    if compact_config().check_cache_consistency:
        compact_cache().check_consistency(group=process_group)
    return out, lse, None