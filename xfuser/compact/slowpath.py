import torch
import torch.distributed as dist
from pipefuser.compact.compress_topk import (
    topk_compress,
    topk_decompress,
    sim_topk,
    SPARSE_LAST_DIM_SIZE,
)
from pipefuser.compact.compress_quantize import (
    quantize_1bit,
    dequantize_1bit,
    quantize_int2,
    dequantize_int2,
    sim_binary,
    sim_int2,
)
from pipefuser.compact.compress_lowrank import (
    subspace_iter,
)
from pipefuser.compact.utils import (
    CompactConfig,
    CompactCache,
    COMPACT_COMPRESS_TYPE,
    PowerCache,
)
def slowpath_compress(x: torch.Tensor, compress_type: COMPACT_COMPRESS_TYPE, rank: int = None, sparse_ratio: int = None):
    """
    Pure function to compress a tensor using the specified method.
    
    Args:
        x: Input tensor to compress. Must be FP16.
        compress_type: The compression type to use.
        rank: The rank parameter for low-rank compression methods.
        sparse_ratio: The sparsity ratio for sparse compression methods.
        
    Returns:
        A compressed tensor.
    """
    assert x.dtype == torch.half
    assert x.dim() == 2
    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        q, scale = quantize_1bit(x)
        assert q.size(0) == scale.size(0)
        # NOTE: Casting all non-half tensors to half
        assert q.dtype == torch.uint8
        assert scale.dtype == torch.half
        q = q.view(torch.half)
        comp_list = [q, scale]
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2:
        q, scale = quantize_int2(x)
        # NOTE: Casting all non-half tensors to half
        assert q.dtype == torch.uint8
        assert scale.dtype == torch.half
        q = q.view(torch.half)
        comp_list = [q, scale]
        assert q.size(1) == scale.size(1) // 4
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        u, v, _ = subspace_iter(x, rank, 2)
        assert u.size(1) == v.size(0) and u.dtype == torch.half and v.dtype == torch.half
        # contiguous() is necessary for later cat
        comp_list = [u.contiguous(), v.contiguous()]
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        val, idx = topk_compress(x.view(-1, SPARSE_LAST_DIM_SIZE), sparse_ratio)
        comp_list = [val, idx.view(torch.half)]
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")
    
    comp_merged = torch.cat([x.view(-1) for x in comp_list], dim=0)
    return comp_merged

def slowpath_decompress(x: torch.Tensor, shape: tuple, compress_type: COMPACT_COMPRESS_TYPE, rank: int = None, sparse_ratio: int = None):
    """
    Pure function to decompress a tensor using the specified method.
    
    Args:
        x: Compressed tensor to decompress. Must be FP16.
        shape: The original shape of the tensor.
        compress_type: The compression type used for compression.
        rank: The rank parameter for low-rank compression methods.
        sparse_ratio: The sparsity ratio for sparse compression methods.
        
    Returns:
        The decompressed tensor with the specified shape.
    """
    assert x.dim() == 1 # NOTE: we previously flattened the cat_list
    channel_size = shape[-1]
    assert len(shape) == 2
    
    assert x.dtype == torch.half
    numel = torch.prod(torch.tensor(shape, dtype=torch.int64)).item()

    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        # q and scale have same count, but q is 1-bit
        split_size = [numel // 16, channel_size]
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2:
        split_size = [numel // 8, channel_size]
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        split_size = [shape[0] * rank, shape[1] * rank]
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        # val and idx have same count, but idx is 4-bit
        split_size = [numel // sparse_ratio, numel // sparse_ratio // 4]
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")

    split_list = torch.split(x, split_size, dim=0)


    if compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        q = split_list[0].view(torch.uint8).view(channel_size, -1)
        scale = split_list[1]
        return dequantize_1bit(q, scale).view(shape)
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2:
        q = split_list[0].view(torch.uint8).view(-1, channel_size//4)
        scale = split_list[1]
        return dequantize_int2(q, scale).view(shape)
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        u = split_list[0].view(shape[0], rank)
        v = split_list[1].view(rank, shape[1])
        return torch.matmul(u, v)
    elif compress_type == COMPACT_COMPRESS_TYPE.SPARSE:
        idx_last_dim_size = SPARSE_LAST_DIM_SIZE//sparse_ratio
        val_last_dim_size = SPARSE_LAST_DIM_SIZE//sparse_ratio//4
        val = split_list[0].view(-1, val_last_dim_size)
        idx = split_list[1].view(torch.uint8).view(-1, idx_last_dim_size)
        return topk_decompress(val, idx, sparse_ratio).view(shape)
    else:
        raise ValueError(f"Invalid compress_type value: {compress_type}")
    
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
        return sim_topk(x, sparse_ratio)
    elif compress_type == COMPACT_COMPRESS_TYPE.BINARY:
        return sim_binary(x)
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2:
        return sim_int2(x)
    elif compress_type == COMPACT_COMPRESS_TYPE.BINARY_SPARSE:
        q = sim_binary(x)
        residual = x - q
        sparse = sim_topk(residual, sparse_ratio)
        return sparse + q
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2_SPARSE:
        q = sim_int2(x)
        residual = x - q
        sparse = sim_topk(residual, sparse_ratio)
        return sparse + q
    elif compress_type == COMPACT_COMPRESS_TYPE.BINARY_LOW_RANK:
        scale = torch.matmul(*subspace_iter(x.abs(), rank, 2))
        return sim_binary(x, scale)
    elif compress_type == COMPACT_COMPRESS_TYPE.INT2_LOW_RANK:
        scale = torch.matmul(*subspace_iter(x.abs(), rank, 2))
        return sim_int2(x, scale)
    elif compress_type == COMPACT_COMPRESS_TYPE.LOW_RANK:
        u, v, q = subspace_iter(x, rank, 2)
        return torch.matmul(u, v)
    else:
        raise ValueError("Invalid compress_type value")