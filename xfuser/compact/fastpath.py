import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler
from xfuser.compact.compress_quantize import dequantize_1bit, quantize_1bit
# from xfuser.compact.compress_quantize import quantize_1bit # Not needed for sim
# Import subspace iter for rank-1 scale approximation
from xfuser.compact.compress_lowrank import subspace_iter

@triton.jit
def _bitwise_or(a, b):
    """ Helper for Triton reduce """
    return a | b

@triton.jit
def _binary_quant_fastpath(
    # Input Pointers (Layout C, N)
    x_ptr,             # Current activation (C, N)
    base_ptr,          # Cached base (C, N)
    delta_base_ptr,    # Cached delta_base (C, N) - Only used if RESIDUAL_LEVEL == 2
    scale_u_ptr,       # Input scale factor u (C,)
    scale_v_ptr,       # Input scale factor v (N,)
    # Output Pointers
    packed_out_ptr,    # Packed delta or delta_delta (C, N//8) - OUTPUT
    new_base_ptr,      # Output new base (C, N) - optional write
    new_delta_base_ptr,# Output new delta_base (C, N) - optional write (only if RESIDUAL_LEVEL == 2)
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    # Strides (ALL are for C, N layout, except packed and scale)
    stride_xc, stride_xn,
    stride_bc, stride_bn,
    stride_dbc, stride_dbn, # Only used if RESIDUAL_LEVEL == 2
    stride_scale_u, # stride for u (C,)
    stride_scale_v, # stride for v (N,)
    stride_packed_c, stride_packed_n8, # Strides for packed output (C, N//8)
    stride_newb_c, stride_newb_n,      # Strides for new_base (C, N)
    stride_newdb_c, stride_newdb_n,    # Strides for new_delta_base (C, N) - Only used if RESIDUAL_LEVEL == 2
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
    UPDATE_CACHE: tl.constexpr,
    DELTA_DECAY: tl.constexpr, # Decay factor for delta_base
    RESIDUAL_LEVEL: tl.constexpr, # Added: 1 or 2
):
    """
    Quantizes delta (level 1) or delta_delta (level 2).
    Packs into 1-bit representation using sign. Scale is pre-calculated based on abs(delta) or abs(delta_delta).
    Optionally updates cache based on RESIDUAL_LEVEL.
    Grid: (CHANNEL, cdiv(N_TOKENS, BLOCK_SIZE_N))
    """
    pid_c = tl.program_id(0); pid_n_block = tl.program_id(1)
    n_block_start = pid_n_block * BLOCK_SIZE_N
    offs_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N_TOKENS

    # --- Load Inputs ---
    x_row_ptr = x_ptr + pid_c * stride_xc
    base_row_ptr = base_ptr + pid_c * stride_bc
    x_block = tl.load(x_row_ptr + offs_n * stride_xn, mask=mask_n, other=0.0)
    base_block = tl.load(base_row_ptr + offs_n * stride_bn, mask=mask_n, other=0.0)

    # --- Calculate Tensor to Quantize ---
    tensor_to_quantize = None
    db_block = None # Initialize db_block
    if RESIDUAL_LEVEL == 1:
        tensor_to_quantize = x_block - base_block
    elif RESIDUAL_LEVEL == 2:
        db_row_ptr = delta_base_ptr + pid_c * stride_dbc
        db_block = tl.load(db_row_ptr + offs_n * stride_dbn, mask=mask_n, other=0.0)
        tensor_to_quantize = x_block - base_block - db_block


    # --- Quantize (Pack Signs) ---
    binary = (tensor_to_quantize >= 0).to(tl.uint8)
    binary_reshaped = tl.reshape(binary, (BLOCK_SIZE_N // 8, 8))
    shifts = tl.arange(0, 8).to(tl.uint8)
    shifted = (binary_reshaped << shifts).to(tl.uint8)
    packed_block = tl.reduce(shifted, axis=1, combine_fn=_bitwise_or).to(tl.uint8)

    # --- Store Packed Block ---
    n8_block_start = n_block_start // 8
    offs_n8 = n8_block_start + tl.arange(0, BLOCK_SIZE_N // 8)
    mask_n8 = offs_n8 < N_TOKENS_8
    packed_output_base_ptr = packed_out_ptr + pid_c * stride_packed_c
    packed_output_ptrs = packed_output_base_ptr + offs_n8 * stride_packed_n8
    tl.store(packed_output_ptrs, packed_block, mask=mask_n8)

    # --- Update Cache (Conditional) ---
    if UPDATE_CACHE:
        # Load scale components
        scale_u_val = tl.load(scale_u_ptr + pid_c * stride_scale_u).to(tl.float16)
        scale_v_block = tl.load(scale_v_ptr + offs_n * stride_scale_v, mask=mask_n, other=0.0).to(tl.float16)
        scale_block = scale_u_val * scale_v_block

        # Dequantize based on the `binary` block
        sign_int8 = tl.where(mask_n, (2 * binary.to(tl.int8) - 1), 0)
        recv_quantized_block = sign_int8 * scale_block # This is recv_delta or recv_delta_delta

        # Calculate new base and potentially new delta_base
        new_base_block = None
        new_delta_base_block = None

        if RESIDUAL_LEVEL == 1:
            new_base_block = base_block + recv_quantized_block # new_base = base + recv_delta
            # new_delta_base is not updated for level 1
        elif RESIDUAL_LEVEL == 2:
            # Need db_block loaded earlier
            new_base_block = base_block + db_block + recv_quantized_block # new_base = base + db + recv_dd
            new_delta_base_block = (db_block + recv_quantized_block) * DELTA_DECAY # new_db = (db + recv_dd) * decay

        # Store new base
        new_base_row_ptr = new_base_ptr + pid_c * stride_newb_c
        tl.store(new_base_row_ptr + offs_n * stride_newb_n, new_base_block, mask=mask_n)

        # Store new delta_base only if level 2
        if RESIDUAL_LEVEL == 2:
            new_db_row_ptr = new_delta_base_ptr + pid_c * stride_newdb_c
            tl.store(new_db_row_ptr + offs_n * stride_newdb_n, new_delta_base_block, mask=mask_n)


@Profiler.prof_func("compact.binary_quant_fastpath")
def binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    delta_base_tensor_cn: torch.Tensor | None, # Input (C, N), None if residual_level=1
    update_cache: bool,
    delta_decay_factor: float,
    residual_level: int, # Added: 1 or 2
):
    """
    Quantizes delta (level 1) or delta_delta (level 2) to 1-bit using fast path kernel.
    Scale is calculated based on abs(tensor_to_quantize).
    Returns: packed, scale_u(C), scale_v(N), new_base, new_delta_base (None if level 1 or not update_cache)
    """
    # Assertions
    assert residual_level in [1, 2], "Residual level must be 1 or 2 for fastpath"
    assert x_tensor_cn.dtype == torch.half
    assert base_tensor_cn.dtype == torch.half
    assert x_tensor_cn.ndim == 2 and base_tensor_cn.ndim == 2
    assert x_tensor_cn.shape == base_tensor_cn.shape
    assert x_tensor_cn.is_cuda and base_tensor_cn.is_cuda
    if residual_level == 2:
        assert delta_base_tensor_cn is not None, "delta_base_tensor_cn must be provided for residual_level=2"
        assert delta_base_tensor_cn.dtype == torch.half
        assert delta_base_tensor_cn.ndim == 2 and delta_base_tensor_cn.shape == x_tensor_cn.shape
        assert delta_base_tensor_cn.is_cuda
    else: # residual_level == 1
        assert delta_base_tensor_cn is None, "delta_base_tensor_cn must be None for residual_level=1"


    x_tensor_cn = x_tensor_cn.contiguous()
    base_tensor_cn = base_tensor_cn.contiguous()
    if residual_level == 2:
        delta_base_tensor_cn = delta_base_tensor_cn.contiguous()

    CHANNEL, N_TOKENS = x_tensor_cn.shape
    assert N_TOKENS % 8 == 0, "N_TOKENS must be divisible by 8 for packing output alignment"
    N_TOKENS_8 = N_TOKENS // 8

    # Calculate tensor to quantize based on residual level
    tensor_to_quantize_cn = None
    if residual_level == 1:
        tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn
    elif residual_level == 2:
        tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn - delta_base_tensor_cn

    # Calculate rank-1 approximation for scale based on abs(tensor_to_quantize)
    with Profiler.scope("compact.quant.scale_rank1_approx"):
        scale_U, scale_V_t, _ = subspace_iter(
            torch.abs(tensor_to_quantize_cn), rank=1, num_iters=2
        )
    scale_u_output = scale_U.squeeze(-1).contiguous().to(torch.half) # Shape (C,)
    scale_v_output = scale_V_t.squeeze(0).contiguous().to(torch.half) # Shape (N,)
    assert scale_u_output.shape == (CHANNEL,) and scale_v_output.shape == (N_TOKENS,)

    # Allocate outputs
    packed_output = torch.empty((CHANNEL, N_TOKENS_8), dtype=torch.uint8, device=x_tensor_cn.device)
    new_base_output_cn = torch.empty_like(x_tensor_cn) if update_cache else None
    # Allocate new_delta_base only if needed
    new_delta_base_output_cn = torch.empty_like(x_tensor_cn) if update_cache and residual_level == 2 else None

    BLOCK_SIZE_N = 512
    assert BLOCK_SIZE_N % 8 == 0, "BLOCK_SIZE_N must be divisible by 8"
    grid = (CHANNEL, triton.cdiv(N_TOKENS, BLOCK_SIZE_N))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = x_tensor_cn # Use existing tensor for properties

    # Delta base pointers/strides (dummy if level 1)
    delta_base_ptr = delta_base_tensor_cn if residual_level == 2 else dummy_tensor
    stride_dbc = delta_base_ptr.stride(0) if residual_level == 2 else 0
    stride_dbn = delta_base_ptr.stride(1) if residual_level == 2 else 0

    # New base pointers/strides (dummy if not update_cache)
    new_base_ptr = new_base_output_cn if update_cache else dummy_tensor
    stride_newb_c = new_base_ptr.stride(0) if update_cache else 0
    stride_newb_n = new_base_ptr.stride(1) if update_cache else 0

    # New delta base pointers/strides (dummy if not update_cache OR level 1)
    new_delta_base_ptr = new_delta_base_output_cn if update_cache and residual_level == 2 else dummy_tensor
    stride_newdb_c = new_delta_base_ptr.stride(0) if update_cache and residual_level == 2 else 0
    stride_newdb_n = new_delta_base_ptr.stride(1) if update_cache and residual_level == 2 else 0


    with Profiler.scope("compact._binary_quant_fastpath"):
         _binary_quant_fastpath[grid](
             x_tensor_cn, base_tensor_cn, delta_base_ptr, # Pass potentially dummy delta_base
             scale_u_output, scale_v_output,
             packed_output,
             new_base_ptr,
             new_delta_base_ptr, # Pass potentially dummy new_delta_base
             # --- Dimensions ---
             N_TOKENS, CHANNEL, N_TOKENS_8,
             # --- Strides ---
             x_tensor_cn.stride(0), x_tensor_cn.stride(1),
             base_tensor_cn.stride(0), base_tensor_cn.stride(1),
             stride_dbc, stride_dbn, # Use potentially dummy strides
             scale_u_output.stride(0), # Stride for u (C,)
             scale_v_output.stride(0), # Stride for v (N,) - assume contiguous
             packed_output.stride(0), packed_output.stride(1),
             stride_newb_c, stride_newb_n, # Use potentially dummy strides
             stride_newdb_c, stride_newdb_n, # Use potentially dummy strides
             # --- Meta-parameters ---
             BLOCK_SIZE_N=BLOCK_SIZE_N, # Pass block size
             UPDATE_CACHE=update_cache,
             DELTA_DECAY=float(delta_decay_factor),
             RESIDUAL_LEVEL=residual_level, # Pass residual level
         )

    # Return values based on update_cache flag
    if update_cache:
        # new_delta_base_output_cn will be None if residual_level=1
        return packed_output, scale_u_output, scale_v_output, new_base_output_cn, new_delta_base_output_cn
    else:
        # Always return 5 values, but last two are None if not update_cache
        return packed_output, scale_u_output, scale_v_output, None, None

# Simulation function needs update if used for residual=1 fastpath testing
# For now, assume it's only used for residual=2 comparison
def sim_binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    delta_base_tensor_cn: torch.Tensor | None, # Input (C, N) - Required for Level 2
    update_cache: bool,
    delta_decay_factor: float, # Only used if residual_level=2
    residual_level: int,
):
    """
    Simulated version of binary_quant_fastpath using quantize_1bit and dequantize_1bit.
    Handles both RESIDUAL LEVEL 1 and 2.
    """
    assert residual_level in [1, 2], "Residual level must be 1 or 2"
    if residual_level == 2:
        assert delta_base_tensor_cn is not None, "delta_base_tensor_cn must be provided for residual_level=2"
    else: # residual_level == 1
        assert delta_base_tensor_cn is None, "delta_base_tensor_cn must be None for residual_level=1"

    C, N = x_tensor_cn.shape
    new_base_cn = None
    new_delta_base_cn = None

    if residual_level == 1:
        # Calculate delta (Level 1)
        tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn
        tensor_to_quantize_nc = tensor_to_quantize_cn.transpose(0, 1).contiguous()

        # Quantize delta
        packed_sim, v_n_sim, u_c_sim = quantize_1bit(tensor_to_quantize_nc.to(torch.float16))
        u_c_output, v_n_output = u_c_sim, v_n_sim # Match fastpath output convention

        if update_cache:
            # Dequantize delta
            recv_quantized_nc = dequantize_1bit(packed_sim, v_n_sim, u_c_sim)
            recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)
            # Calculate new base (Level 1)
            new_base_cn = base_tensor_cn + recv_quantized_cn
            # new_delta_base is None for level 1

    else: # residual_level == 2
        # Calculate delta_delta (Level 2)
        tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn - delta_base_tensor_cn
        tensor_to_quantize_nc = tensor_to_quantize_cn.transpose(0, 1).contiguous()

        # Quantize delta_delta
        packed_sim, v_n_sim, u_c_sim = quantize_1bit(tensor_to_quantize_nc.to(torch.float16))
        u_c_output, v_n_output = u_c_sim, v_n_sim # Match fastpath output convention

        if update_cache:
            # Dequantize delta_delta
            recv_quantized_nc = dequantize_1bit(packed_sim, v_n_sim, u_c_sim)
            recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)
            # Calculate new base and new delta_base (Level 2)
            new_base_cn = base_tensor_cn + delta_base_tensor_cn + recv_quantized_cn
            new_delta_base_cn = (delta_base_tensor_cn + recv_quantized_cn) * delta_decay_factor

    # Return values matching fastpath signature: packed, scale_u(C), scale_v(N), new_base, new_delta_base
    return packed_sim, u_c_output, v_n_output, new_base_cn, new_delta_base_cn


@triton.jit
def _binary_dequant_fastpath(
    # Input Pointers
    packed_in_ptr,     # Packed delta or delta_delta (C, N//8) uint8
    scale_u_ptr,       # Scale factor u (C,)
    scale_v_ptr,       # Scale factor v (N,)
    base_ptr,          # Base cache (C, N) half
    delta_base_ptr,    # Delta base cache (C, N) half - Only used if RESIDUAL_LEVEL == 2
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (C, N) half
    new_db_ptr,        # Output new delta_base (C, N) half - Only written if RESIDUAL_LEVEL == 2
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    # Strides
    stride_packed_c, stride_packed_n8,
    stride_scale_u,
    stride_scale_v,
    stride_base_c, stride_base_n,
    stride_db_c, stride_db_n, # Only used if RESIDUAL_LEVEL == 2
    stride_recon_c, stride_recon_n,
    stride_newdb_c, stride_newdb_n, # Only used if RESIDUAL_LEVEL == 2
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
    DELTA_DECAY: tl.constexpr, # Decay factor for delta_base
    RESIDUAL_LEVEL: tl.constexpr, # Added: 1 or 2
):
    """
    Dequantizes delta/delta_delta and calculates reconstructed activation.
    Optionally calculates new_delta_base based on RESIDUAL_LEVEL.
    Grid: (CHANNEL, cdiv(N_TOKENS, BLOCK_SIZE_N))
    """
    pid_c = tl.program_id(0); pid_n_block = tl.program_id(1)
    n_block_start = pid_n_block * BLOCK_SIZE_N
    offs_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N_TOKENS

    # --- Dequantize Block ---
    # Load Scales
    scale_u = tl.load(scale_u_ptr + pid_c * stride_scale_u).to(tl.float16)
    scale_v_block = tl.load(scale_v_ptr + offs_n * stride_scale_v, mask=mask_n, other=0.0).to(tl.float16)
    scale_block = scale_u * scale_v_block

    # Load and unpack bits
    n8_block_start = n_block_start // 8
    offs_n8 = n8_block_start + tl.arange(0, BLOCK_SIZE_N // 8)
    mask_n8 = offs_n8 < N_TOKENS_8
    packed_row_ptr = packed_in_ptr + pid_c * stride_packed_c
    packed_block = tl.load(packed_row_ptr + offs_n8 * stride_packed_n8, mask=mask_n8, other=0)
    byte_indices_in_row = offs_n // 8
    bit_indices_in_byte = offs_n % 8
    # Ensure we only load bytes within the valid packed range (N//8) for the row
    final_byte_mask = mask_n & (byte_indices_in_row < N_TOKENS_8)
    packed_bytes_for_elems = tl.load(packed_row_ptr + byte_indices_in_row * stride_packed_n8, mask=final_byte_mask, other=0)
    bits = ((packed_bytes_for_elems >> bit_indices_in_byte) & 1)
    signs = tl.where(mask_n, (2 * bits - 1).to(tl.int8), 0)
    recv_quantized_block = signs * scale_block # recv_delta or recv_delta_delta

    # --- Load Base and Delta Base (if needed) ---
    base_row_ptr = base_ptr + pid_c * stride_base_c
    base_block = tl.load(base_row_ptr + offs_n * stride_base_n, mask=mask_n, other=0.0)
    db_block = None
    if RESIDUAL_LEVEL == 2:
        db_row_ptr = delta_base_ptr + pid_c * stride_db_c
        db_block = tl.load(db_row_ptr + offs_n * stride_db_n, mask=mask_n, other=0.0)

    # --- Calculate Outputs Block ---
    recon_block = None
    new_db_block = None
    if RESIDUAL_LEVEL == 1:
        recon_block = base_block + recv_quantized_block # recon = base + recv_delta
        # new_db is not calculated or stored
    elif RESIDUAL_LEVEL == 2:
        recon_block = base_block + db_block + recv_quantized_block # recon = base + db + recv_dd
        new_db_block = (db_block + recv_quantized_block) * DELTA_DECAY # new_db = (db + recv_dd) * decay
    else: # Should not happen
        pass

    # --- Store Outputs Block ---
    recon_out_ptr = recon_ptr + pid_c * stride_recon_c
    tl.store(recon_out_ptr + offs_n * stride_recon_n, recon_block, mask=mask_n)
    # Store new delta_base only if level 2
    if RESIDUAL_LEVEL == 2:
        new_db_out_ptr = new_db_ptr + pid_c * stride_newdb_c
        tl.store(new_db_out_ptr + offs_n * stride_newdb_n, new_db_block, mask=mask_n)


@Profiler.prof_func("compact.binary_dequant_fastpath")
def binary_dequant_fastpath(
    packed_in_cn8: torch.Tensor,    # Input packed delta/dd (C, N//8) uint8
    scale_u_c: torch.Tensor,       # Input scale u (C,)
    scale_v_n: torch.Tensor,       # Input scale v (N,)
    base_cn: torch.Tensor,         # Input base cache (C, N) half
    delta_base_cn: torch.Tensor | None, # Input delta_base cache (C, N) half, None if level 1
    delta_decay_factor: float,
    residual_level: int, # Added: 1 or 2
):
    """
    Dequantizes delta/delta_delta and calculates reconstructed activation.
    Optionally calculates and returns new_delta_base based on residual_level.

    Input: packed(C, N//8), u(C), v(N), base(C,N), delta_base(C,N)|None
    Output: reconstructed(C, N), new_delta_base(C, N)|None
    """
    # Assertions
    assert residual_level in [1, 2], "Residual level must be 1 or 2 for fastpath"
    assert packed_in_cn8.dtype == torch.uint8
    assert scale_u_c.dtype == torch.half and scale_v_n.dtype == torch.half
    assert base_cn.dtype == torch.half
    assert packed_in_cn8.ndim == 2 and scale_u_c.ndim == 1 and scale_v_n.ndim == 1 and base_cn.ndim == 2
    if residual_level == 2:
        assert delta_base_cn is not None, "delta_base_cn must be provided for residual_level=2"
        assert delta_base_cn.dtype == torch.half
        assert delta_base_cn.ndim == 2 and delta_base_cn.shape == base_cn.shape
        assert delta_base_cn.is_cuda
    else: # residual_level == 1
        assert delta_base_cn is None, "delta_base_cn must be None for residual_level=1"

    assert packed_in_cn8.is_cuda and scale_u_c.is_cuda and scale_v_n.is_cuda and base_cn.is_cuda

    packed_in_cn8 = packed_in_cn8.contiguous()
    scale_u_c = scale_u_c.contiguous()
    scale_v_n = scale_v_n.contiguous()
    base_cn = base_cn.contiguous()
    if residual_level == 2:
        delta_base_cn = delta_base_cn.contiguous()

    CHANNEL, N_TOKENS_8 = packed_in_cn8.shape
    N_TOKENS = N_TOKENS_8 * 8
    assert base_cn.shape == (CHANNEL, N_TOKENS), f"Base shape mismatch: {base_cn.shape} vs expected {(CHANNEL, N_TOKENS)}"
    assert scale_u_c.shape == (CHANNEL,), f"Scale U shape mismatch: {scale_u_c.shape} vs expected {(CHANNEL,)}"
    assert scale_v_n.shape == (N_TOKENS,), f"Scale V shape mismatch: {scale_v_n.shape} vs expected {(N_TOKENS,)}"

    # Allocate output tensors
    reconstructed_output_cn = torch.empty_like(base_cn)
    # Allocate new_delta_base only if needed
    new_delta_base_output_cn = torch.empty_like(delta_base_cn) if residual_level == 2 else None

    BLOCK_SIZE_N = 512
    assert BLOCK_SIZE_N % 8 == 0, "BLOCK_SIZE_N must be divisible by 8 for unpacking logic"
    grid = (CHANNEL, triton.cdiv(N_TOKENS, BLOCK_SIZE_N))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = base_cn # Use existing tensor for properties

    # Delta base pointers/strides (dummy if level 1)
    delta_base_ptr = delta_base_cn if residual_level == 2 else dummy_tensor
    stride_dbc = delta_base_ptr.stride(0) if residual_level == 2 else 0
    stride_dbn = delta_base_ptr.stride(1) if residual_level == 2 else 0

    # New delta base output pointers/strides (dummy if level 1)
    new_db_ptr = new_delta_base_output_cn if residual_level == 2 else dummy_tensor
    stride_newdb_c = new_db_ptr.stride(0) if residual_level == 2 else 0
    stride_newdb_n = new_db_ptr.stride(1) if residual_level == 2 else 0

    with Profiler.scope("compact._binary_dequant_fastpath"):
         _binary_dequant_fastpath[grid](
             packed_in_cn8,
             scale_u_c, scale_v_n,
             base_cn, delta_base_ptr, # Pass potentially dummy delta_base
             reconstructed_output_cn, new_db_ptr, # Pass potentially dummy new_db output
             # --- Dimensions ---
             N_TOKENS, CHANNEL, N_TOKENS_8,
             # --- Strides ---
             packed_in_cn8.stride(0), packed_in_cn8.stride(1),
             scale_u_c.stride(0),
             scale_v_n.stride(0),
             base_cn.stride(0), base_cn.stride(1),
             stride_dbc, stride_dbn, # Use potentially dummy strides
             reconstructed_output_cn.stride(0), reconstructed_output_cn.stride(1),
             stride_newdb_c, stride_newdb_n, # Use potentially dummy strides
             # --- Meta-parameters ---
             BLOCK_SIZE_N=BLOCK_SIZE_N,
             DELTA_DECAY=float(delta_decay_factor),
             RESIDUAL_LEVEL=residual_level, # Pass residual level
         )

    # Return recon and potentially new_delta_base (None if level 1)
    return reconstructed_output_cn, new_delta_base_output_cn

# Simulation function needs update if used for residual=1 fastpath testing
# For now, assume it's only used for residual=2 comparison
def sim_binary_dequant_fastpath(
    packed_in_cn8: torch.Tensor, # Packed bits (C, N//8) UINT8
    scale_u_c: torch.Tensor,     # Scale factor U (C,) FP16
    scale_v_n: torch.Tensor,     # Scale factor V (N,) FP16
    base_cn: torch.Tensor,
    delta_base_cn: torch.Tensor | None, # Required for Level 2
    delta_decay_factor: float, # Only used if residual_level=2
    residual_level: int,
):
    """
    Simulated version of binary_dequant_fastpath using dequantize_1bit.
    Accepts scales u(C), v(N) consistent with kernel path signature.
    Internally swaps scales when calling dequantize_1bit.
    Handles both RESIDUAL LEVEL 1 and 2.
    """
    assert residual_level in [1, 2], "Residual level must be 1 or 2"
    if residual_level == 2:
        assert delta_base_cn is not None, "delta_base_cn must be provided for residual_level=2"
    else: # residual_level == 1
        assert delta_base_cn is None, "delta_base_cn must be None for residual_level=1"

    C = scale_u_c.shape[0]
    N = scale_v_n.shape[0]
    # dequantize_1bit expects packed(C, N//8), u(N), v(C)
    recv_quantized_nc = dequantize_1bit(packed_in_cn8, scale_v_n, scale_u_c)
    recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)

    reconstructed_cn = None
    new_delta_base_cn = None

    if residual_level == 1:
        # Calculate reconstructed activation (Level 1)
        reconstructed_cn = base_cn + recv_quantized_cn
        # new_delta_base is None for level 1
    else: # residual_level == 2
        # Calculate final outputs (Level 2 logic)
        reconstructed_cn = base_cn + delta_base_cn + recv_quantized_cn
        new_delta_base_cn = (delta_base_cn + recv_quantized_cn) * delta_decay_factor

    return reconstructed_cn, new_delta_base_cn


# --- Test Functions --- 
def profile_quantize_kernels(num_runs=1000, num_warmup=5):
    """Profile quantize kernel for both residual levels."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    DELTA_DECAY_FACTOR = 0.5
    atol, rtol = 1e-3, 1e-2

    print("--- Quant Performance & Correctness --- (ms per run)")

    for level in [1, 2]:
        print(f"\nTesting Residual Level: {level}")
        # Setup Tensors
        x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
        base_tensor_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()
        delta_base_tensor_cn_l2 = (torch.randn_like(x_tensor_cn) * 0.05).contiguous() if level == 2 else None

        # Kernel arg for delta_base
        kernel_delta_base_arg = delta_base_tensor_cn_l2
        # Sim args
        sim_args = (x_tensor_cn, base_tensor_cn, delta_base_tensor_cn_l2, True, DELTA_DECAY_FACTOR, level)

        # --- Calculate Reference Values (Simulation) ---
        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            ref_packed, ref_scale_u, ref_scale_v, ref_new_base, ref_new_db = sim_binary_quant_fastpath(*sim_args)

        # --- Warm-up runs ---
        for _ in range(num_warmup):
            with torch.random.fork_rng(devices=['cuda']):
                 torch.manual_seed(42)
                 _ = binary_quant_fastpath(
                     x_tensor_cn, base_tensor_cn, kernel_delta_base_arg,
                     update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
                 )
                 _ = sim_binary_quant_fastpath(*sim_args) # Use updated sim call
            torch.cuda.synchronize()

        # --- Get Kernel results for correctness check ---
        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            kernel_packed, kernel_scale_u, kernel_scale_v, kernel_new_base, kernel_new_db = binary_quant_fastpath(
                x_tensor_cn, base_tensor_cn, kernel_delta_base_arg,
                update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
            )
        torch.cuda.synchronize()

        # --- Profiling Kernel ---
        torch.cuda.synchronize()
        start_kernel = time.time()
        for i in range(num_runs):
            update_c = (i % 2 == 0)
            with torch.random.fork_rng(devices=['cuda']):
                 torch.manual_seed(42 + i)
                 _ = binary_quant_fastpath(
                     x_tensor_cn, base_tensor_cn, kernel_delta_base_arg,
                     update_cache=update_c, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
                 )
        torch.cuda.synchronize()
        end_kernel = time.time()
        kernel_time = (end_kernel - start_kernel) / num_runs
        print(f"Kernel (Level {level}): {kernel_time*1000:.3f} ms")

        # --- Verify correctness ---
        correct = True
        issues = []
        # Packed
        if not torch.equal(ref_packed, kernel_packed):
            correct = False; issues.append("Packed")
        # Scale U
        if not torch.allclose(ref_scale_u, kernel_scale_u, atol=atol, rtol=rtol):
            correct = False; issues.append(f"Scale U (Max Diff: {torch.max(torch.abs(ref_scale_u - kernel_scale_u))})")
        # Scale V
        if not torch.allclose(ref_scale_v, kernel_scale_v, atol=atol, rtol=rtol):
             correct = False; issues.append(f"Scale V (Max Diff: {torch.max(torch.abs(ref_scale_v - kernel_scale_v))})")
        # New Base
        if ref_new_base is not None and kernel_new_base is not None:
             if not torch.allclose(ref_new_base, kernel_new_base, atol=atol, rtol=rtol):
                 correct = False; issues.append(f"New Base (Max Diff: {torch.max(torch.abs(ref_new_base - kernel_new_base))})")
        elif ref_new_base is not None or kernel_new_base is not None:
             correct = False; issues.append("New Base (Mismatch None)")
        # New Delta Base
        if ref_new_db is not None and kernel_new_db is not None:
            if not torch.allclose(ref_new_db, kernel_new_db, atol=atol, rtol=rtol):
                 correct = False; issues.append(f"New Delta Base (Max Diff: {torch.max(torch.abs(ref_new_db - kernel_new_db))})")
        elif ref_new_db is not None or kernel_new_db is not None:
             correct = False; issues.append("New Delta Base (Mismatch None)")

        print(f"Correctness vs Sim (Level {level}): {correct}")
        if not correct: print(f"  Issues: {', '.join(issues)}")
    print("---")


def profile_dequantize_kernels(num_runs=1000, num_warmup=5):
    """Profile dequantize kernel for both residual levels."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    DELTA_DECAY_FACTOR = 0.5
    atol, rtol = 1e-3, 1e-2

    print("--- Dequant Performance & Correctness --- (ms per run)")

    for level in [1, 2]:
        print(f"\nTesting Residual Level: {level}")
        # Setup Tensors
        x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
        base_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()
        delta_base_cn_l2 = (torch.randn_like(x_tensor_cn) * 0.05).contiguous() if level == 2 else None

        # --- Generate Inputs using Quant Sim ---
        quant_sim_args = (x_tensor_cn, base_cn, delta_base_cn_l2, False, DELTA_DECAY_FACTOR, level)

        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            input_packed, input_scale_u, input_scale_v, _, _ = sim_binary_quant_fastpath(*quant_sim_args)

        # --- Calculate Reference Outputs (Dequant Simulation) ---
        dequant_sim_args = (input_packed, input_scale_u, input_scale_v, base_cn, delta_base_cn_l2, DELTA_DECAY_FACTOR, level)

        ref_reconstructed_cn, ref_new_delta_base_cn = sim_binary_dequant_fastpath(*dequant_sim_args)

        # Select kernel args
        kernel_delta_base_arg = delta_base_cn_l2

        # --- Warm-up runs ---
        for _ in range(num_warmup):
            _ = binary_dequant_fastpath(
                input_packed, input_scale_u, input_scale_v,
                base_cn, kernel_delta_base_arg, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
            )
            _ = sim_binary_dequant_fastpath(*dequant_sim_args) # Use updated sim call
            torch.cuda.synchronize()

        # --- Get Kernel results for correctness check ---
        kernel_reconstructed_cn, kernel_new_delta_base_cn = binary_dequant_fastpath(
            input_packed, input_scale_u, input_scale_v,
            base_cn, kernel_delta_base_arg, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
        )
        torch.cuda.synchronize()

        # --- Profiling Kernel ---
        torch.cuda.synchronize()
        start_kernel = time.time()
        for _ in range(num_runs):
            _ = binary_dequant_fastpath(
                input_packed, input_scale_u, input_scale_v,
                base_cn, kernel_delta_base_arg, delta_decay_factor=DELTA_DECAY_FACTOR, residual_level=level
            )
        torch.cuda.synchronize()
        end_kernel = time.time()
        kernel_time = (end_kernel - start_kernel) / num_runs
        print(f"Kernel (Level {level}): {kernel_time*1000:.3f} ms")

        # --- Verify correctness ---
        correct = True
        issues = []
        # Reconstructed
        if not torch.allclose(ref_reconstructed_cn, kernel_reconstructed_cn, atol=atol, rtol=rtol):
            correct = False; issues.append(f"Reconstructed (Max Diff: {torch.max(torch.abs(ref_reconstructed_cn - kernel_reconstructed_cn))})")
        # New Delta Base
        if ref_new_delta_base_cn is not None and kernel_new_delta_base_cn is not None:
            if not torch.allclose(ref_new_delta_base_cn, kernel_new_delta_base_cn, atol=atol, rtol=rtol):
                 correct = False; issues.append(f"New Delta Base (Max Diff: {torch.max(torch.abs(ref_new_delta_base_cn - kernel_new_delta_base_cn))})")
        elif ref_new_delta_base_cn is not None or kernel_new_delta_base_cn is not None:
            # This check is important: kernel should return None for level 1
             correct = False; issues.append("New Delta Base (Mismatch None)")

        print(f"Correctness vs Sim (Level {level}): {correct}")
        if not correct: print(f"  Issues: {', '.join(issues)}")
    print("---")


if __name__ == "__main__":
    # Test functions should now profile and check correctness for both levels
    profile_quantize_kernels()
    profile_dequantize_kernels()
