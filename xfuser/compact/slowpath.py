import torch
import torch.distributed as dist
from xfuser.compact.compress_topk import (
    topk_compress,
    topk_decompress,
    sim_topk,
    SPARSE_LAST_DIM_SIZE,
)
from xfuser.compact.compress_quantize import (
    quantize_1bit,
    dequantize_1bit,
    sim_binary,
    sim_int2,
)
from xfuser.compact.compress_lowrank import (
    subspace_iter,
)
from xfuser.compact.utils import (
    CompactConfig,
    CompactCache,
    COMPACT_COMPRESS_TYPE,
)

def slowpath_compress(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, rank: int = None, sparse_ratio: int = None):
    """
    Pure function to compress a tensor using the specified method.
    Input layout: (N, C)
    
    Args:
        x: Input tensor to compress. Must be FP16.
        compress_type: The compression type to use.
        rank: The rank parameter for low-rank or binary compression methods.
        sparse_ratio: The sparsity ratio for sparse compression methods.
        
    Returns:
        A compressed tensor.
    """
    assert x.dtype == torch.half, f"x.dtype: {x.dtype}"
    assert x.dim() == 2
    N, C = x.shape

    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        assert rank is not None and (rank >= 1 or rank == -1), "Rank must be >= 1 or -1 for BINARY compression"
        q, scale_u, scale_v = quantize_1bit(x, rank=rank) # Pass rank
        assert q.dtype == torch.uint8
        effective_rank = 1 if rank == -1 else rank
        assert q.shape == (N, C // 8) # Check new packed shape
        assert scale_u.dtype == torch.half and scale_u.shape == (N, effective_rank)
        assert scale_v.dtype == torch.half and scale_v.shape == (effective_rank, C)
        # Flatten u and v for concatenation
        comp_list = [q.view(torch.half).contiguous(), scale_u.view(-1).contiguous(), scale_v.view(-1).contiguous()]
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        assert rank is not None and rank >= 1, "Rank must be provided for LOW_RANK compression"
        # assert shape
        u, v, _ = subspace_iter(x, rank, 2)
        u = u.half()
        v = v.half()
        assert u.size(1) == v.size(0) and u.dtype == torch.half and v.dtype == torch.half
        # contiguous() is necessary for later cat
        comp_list = [u.contiguous(), v.contiguous()]
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        assert sparse_ratio is not None, "sparse_ratio must be provided for SPARSE compression"
        val, idx = topk_compress(x.view(-1, SPARSE_LAST_DIM_SIZE), sparse_ratio)
        comp_list = [val, idx.view(torch.half)]
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")
    
    comp_merged = torch.cat([x.view(-1) for x in comp_list], dim=0)
    return comp_merged

def slowpath_decompress(x: torch.Tensor, shape: tuple, compress_type: COMPACT_COMPRESS_TYPE, rank: int = None, sparse_ratio: int = None):
    """
    Pure function to decompress a tensor using the specified method.
    Output layout: (N, C)
    
    Args:
        x: Compressed tensor to decompress. Must be FP16.
        shape: The original shape of the tensor.
        compress_type: The compression type used for compression.
        rank: The rank parameter for low-rank or binary compression methods.
        sparse_ratio: The sparsity ratio for sparse compression methods.
        
    Returns:
        The decompressed tensor with the specified shape.
    """
    assert x.dim() == 1 # NOTE: we previously flattened the cat_list
    assert len(shape) == 2
    N, C = shape # Get N and C explicitly
    numel = N * C
    
    assert x.dtype == torch.half

    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        assert rank is not None and (rank >= 1 or rank == -1), "Rank must be >= 1 or -1 for BINARY decompression"
        effective_rank = 1 if rank == -1 else rank # Determine effective rank
        q_numel_uint8 = numel // 8 # Number of UINT8 elements for q
        q_numel_half = q_numel_uint8 // 2 # Number of FP16 elements for q (packed)
        u_numel_half = N * effective_rank       # Number of FP16 elements for scale_u
        v_numel_half = effective_rank * C       # Number of FP16 elements for scale_v
        # Split sizes are in terms of FP16 elements
        split_size = [q_numel_half, u_numel_half, v_numel_half]
        # Check calculation against actual compressed size
        assert sum(split_size) == x.numel(), f"Binary split error. Calculated sum {sum(split_size)} != Actual size {x.numel()}. Shape: {shape}, Rank: {rank} (Eff: {effective_rank}), qN_h: {q_numel_half}, uN_h: {u_numel_half}, vN_h: {v_numel_half}"
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        assert rank is not None and rank >= 1, "Rank must be provided for LOW_RANK decompression"
        split_size = [N * rank, rank * C] # Correct shape for u(N,K) and v(K,C)
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        # val and idx have same count, but idx is 4-bit
        split_size = [numel // sparse_ratio, numel // sparse_ratio // 4]
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")

    split_list = torch.split(x, split_size, dim=0)

    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        q_half = split_list[0]
        scale_u_flat = split_list[1]
        scale_v_flat = split_list[2]
        # Reshape q from FP16 view -> UINT8 view -> (N, C//8)
        q = q_half.view(torch.uint8).view(N, C // 8)
        # Reshape scales to (N, K) and (K, C)
        effective_rank = 1 if rank == -1 else rank # Redetermine for reshaping
        scale_u = scale_u_flat.view(N, effective_rank)
        scale_v = scale_v_flat.view(effective_rank, C)
        # dequantize_1bit now returns (N, C) directly
        return dequantize_1bit(q, scale_u, scale_v)
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        u = split_list[0].view(N, rank) # Reshape to (N, K)
        v = split_list[1].view(rank, C) # Reshape to (K, C)
        return torch.matmul(u, v)
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        idx_last_dim_size = SPARSE_LAST_DIM_SIZE//sparse_ratio
        val_last_dim_size = SPARSE_LAST_DIM_SIZE//sparse_ratio//4
        val = split_list[0].view(-1, val_last_dim_size)
        idx = split_list[1].view(torch.uint8).view(-1, idx_last_dim_size)
        return topk_decompress(val, idx, sparse_ratio).view(shape)
    elif compress_type == COMPACT_COMPRESS_TYPE.BINARY_MEAN_AS_SCALE:
        # This type explicitly uses mean, equivalent to rank=-1
        return sim_binary(x, rank=-1)
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")

_current_lowrank_scale_k = None #size (C,) or (N,)
_current_lowrank_scale_v = None #size (C,) or (N,)

def set_current_lowrank_scale(scale_k, scale_v):
    global _current_lowrank_scale_k, _current_lowrank_scale_v
    _current_lowrank_scale_k = scale_k
    _current_lowrank_scale_v = scale_v

def sim_compress(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, sparse_ratio: int = None, rank: int = None):
    """
    Simulate the compression and decompression of a tensor using the specified method.
    Make it a pure function for testing.
    """
    # if compress_type == COMPACT_COMPRESS_TYPE.WARMUP:
    #     return x
    if compress_type == COMPACT_COMPRESS_TYPE.IDENTITY:
        return x
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        assert sparse_ratio is not None
        return sim_topk(x, sparse_ratio)
    elif compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        assert rank is not None
        quant_x = sim_binary(x.half(), rank=rank)
        return quant_x.half()
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2:
        return sim_int2(x)
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        assert rank is not None
        from xfuser.compact.main import compact_get_current_cache_key
        N, C = x.shape
        cache_key = compact_get_current_cache_key()
        is_k = cache_key.split("-")[-1] == 'k'
        current_lowrank_scale = _current_lowrank_scale_k if is_k else _current_lowrank_scale_v
        
        if current_lowrank_scale is not None:
            if current_lowrank_scale.shape == (C,):
                x = x.float() * current_lowrank_scale.view(1, C)
            elif current_lowrank_scale.shape == (N,):
                x = x.float() * current_lowrank_scale.view(N, 1)
        u, v, _ = subspace_iter(x, rank, 2)
        if current_lowrank_scale is not None:
            if current_lowrank_scale.shape == (C,):
                v = v / current_lowrank_scale.view(1, C)
            elif current_lowrank_scale.shape == (N,):
                u = u / current_lowrank_scale.view(N, 1)
        # from xfuser.compact.compress_quantize import sim_int4
        # import math
        # from fast_hadamard_transform import hadamard_transform
        # N = u.shape[1]
        # def next_power_of_two(n: int):
        #     return 1 << (n - 1).bit_length()
        # N = next_power_of_two(N)
        # scale = 1 / math.sqrt(N)
        # u_h = hadamard_transform(u, scale=scale)
        # v_h = hadamard_transform(v.t(), scale=scale).t()
        # assert u_h.shape == u.shape
        # assert v_h.shape == v.shape
        # u = u_h
        # v = v_h
        # u = sim_int2(u_h)
        # v = sim_int2(v_h.t()).t()
        return torch.matmul(u, v)
    else:
        raise ValueError("Invalid compress_type value")