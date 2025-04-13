import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler
from xfuser.compact.compress_quantize import dequantize_1bit
from xfuser.compact.compress_quantize import quantize_1bit
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
    scale_u_ptr,       # Input scale factor u (C, K or C, 1)
    scale_v_ptr,       # Input scale factor v (K, N or 1, N)
    # Output Pointers
    packed_out_ptr,    # Packed delta (C, N//8) - OUTPUT
    new_base_ptr,      # Output new base (C, N) - optional write
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    RANK: tl.constexpr,      # Effective rank (1 for mean, K for subspace)
    # Strides (ALL are for C, N layout, except packed and scale)
    stride_xc, stride_xn,
    stride_bc, stride_bn,
    stride_scale_uc, stride_scale_uk, # Stride for u (C, K or C, 1)
    stride_scale_vn, stride_scale_vk, # Stride for v (K, N or 1, N)
    stride_packed_c, stride_packed_n8, # Strides for packed output (C, N//8)
    stride_newb_c, stride_newb_n,      # Strides for new_base (C, N)
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
    UPDATE_CACHE: tl.constexpr,
):
    """
    Quantizes delta (x - base) to 1-bit.
    Packs into 1-bit representation using sign. Scale is calculated using rank-K or rank-1 approximation INSIDE kernel.
    Optionally updates base cache.
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

    # --- Calculate Tensor to Quantize (Always delta) ---
    tensor_to_quantize = x_block - base_block

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
        # --- Load Scale Components (Rank-K or Rank-1) ---
        offs_k = tl.arange(0, RANK)
        # Load U vector for current channel pid_c: shape (K,) or (1,)
        scale_u_ptr_base = scale_u_ptr + pid_c * stride_scale_uc
        scale_u_vec = tl.load(scale_u_ptr_base + offs_k * stride_scale_uk) # (RANK,)

        # Load V block for current N block: shape (RANK, BLOCK_SIZE_N) from V (K, N or 1, N)
        offs_n_masked = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        # Correct broadcasting for V offsets: (K, 1) for k_offsets, (1, BLOCK_SIZE_N) for n_offsets
        # V layout is (K, N), stride_scale_vk = stride K dim, stride_scale_vn = stride N dim
        offs_v = offs_k[:, None] * stride_scale_vk + offs_n_masked[None, :] * stride_scale_vn # <<< CHANGED Calculation
        mask_v = mask_n[None, :] & (offs_k[:, None] < RANK) # Shape (RANK, BLOCK_SIZE_N)
        scale_v_block = tl.load(scale_v_ptr + offs_v, mask=mask_v, other=0.0) # (RANK, BLOCK_SIZE_N)

        # --- Calculate Scale (Rank-K or Rank-1 dot product) ---
        # scale = dot(u_k, v_k) -> sum(u_k * v_k)
        # If RANK=1, this is just u[0] * v[0, :] which is mean_scale * 1.0
        scale_block = tl.sum(scale_u_vec[:, None] * scale_v_block, axis=0).to(tl.float16) # (BLOCK_SIZE_N,)

        # Dequantize based on the `binary` block
        sign_int8 = tl.where(mask_n, (2 * binary.to(tl.int8) - 1), 0)
        recv_quantized_block = sign_int8 * scale_block # This is recv_delta

        # Calculate new base
        new_base_block = base_block + recv_quantized_block # new_base = base + recv_delta

        # Store new base
        new_base_row_ptr = new_base_ptr + pid_c * stride_newb_c
        tl.store(new_base_row_ptr + offs_n * stride_newb_n, new_base_block, mask=mask_n)


@Profiler.prof_func("compact.binary_quant_fastpath")
def binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    rank: int,                        # Rank K or -1 for channel-wise mean
    update_cache: bool,
):
    """
    Quantizes delta (x - base) to 1-bit using fast path kernel.
    Calculates Rank-K scale factors U(C,K), V(K,N) OR channel-wise mean scale (if rank=-1).
    Passes scales to kernel for internal scale calculation.
    Returns: packed(C, N//8), scale_u(C,K or C,1), scale_v(K,N or 1,N), new_base(C,N)|None
    """
    # Assertions
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"
    assert x_tensor_cn.dtype == torch.half
    assert base_tensor_cn.dtype == torch.half
    assert x_tensor_cn.ndim == 2 and base_tensor_cn.ndim == 2
    assert x_tensor_cn.shape == base_tensor_cn.shape
    assert x_tensor_cn.is_cuda and base_tensor_cn.is_cuda


    x_tensor_cn = x_tensor_cn.contiguous()
    base_tensor_cn = base_tensor_cn.contiguous()

    CHANNEL, N_TOKENS = x_tensor_cn.shape
    assert N_TOKENS % 8 == 0, "N_TOKENS must be divisible by 8 for packing output alignment"
    N_TOKENS_8 = N_TOKENS // 8

    # Calculate tensor to quantize (Always delta)
    tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn

    # --- Scale Calculation ---
    if rank == -1:
        # Calculate channel-wise mean scale
        with Profiler.scope("compact.quant.scale_channel_mean"):
            # Scale is the mean of absolute values per channel
            mean_scale_c1 = torch.mean(torch.abs(tensor_to_quantize_cn), dim=1, keepdim=True)
        # Prepare rank-1 structure for the kernel: U = mean_scale, V = ones
        scale_u_output_ck = mean_scale_c1.contiguous().to(torch.half) # Shape (C, 1)
        scale_v_output_kn = torch.ones((1, N_TOKENS), device=x_tensor_cn.device, dtype=torch.half) # Shape (1, N)
        effective_rank = 1 # Kernel needs rank >= 1 for its loop
    else:
        # Calculate rank-K approximation for scale based on abs(tensor_to_quantize)
        with Profiler.scope(f"compact.quant.scale_rank{rank}_approx"):
            scale_U_ck, scale_V_t_kn, _ = subspace_iter(
                torch.abs(tensor_to_quantize_cn), rank=rank, num_iters=2
            )
        # Kernel expects U(C, K) and V(K, N)
        scale_u_output_ck = scale_U_ck.contiguous().to(torch.half)       # Shape (C, K)
        scale_v_output_kn = scale_V_t_kn.contiguous().to(torch.half) # Shape (K, N)
        effective_rank = rank
        assert scale_u_output_ck.shape == (CHANNEL, rank)
        assert scale_v_output_kn.shape == (rank, N_TOKENS)


    # Allocate outputs
    packed_output = torch.empty((CHANNEL, N_TOKENS_8), dtype=torch.uint8, device=x_tensor_cn.device)
    new_base_output_cn = torch.empty_like(x_tensor_cn) if update_cache else None

    BLOCK_SIZE_N = 512
    assert BLOCK_SIZE_N % 8 == 0, "BLOCK_SIZE_N must be divisible by 8"
    grid = (CHANNEL, triton.cdiv(N_TOKENS, BLOCK_SIZE_N))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = x_tensor_cn # Use existing tensor for properties

    # New base pointers/strides (dummy if not update_cache)
    new_base_ptr = new_base_output_cn if update_cache else dummy_tensor
    stride_newb_c = new_base_ptr.stride(0) if update_cache else 0
    stride_newb_n = new_base_ptr.stride(1) if update_cache else 0


    with Profiler.scope("compact._binary_quant_fastpath"):
         _binary_quant_fastpath[grid](
             x_tensor_cn, base_tensor_cn,
             scale_u_output_ck, scale_v_output_kn, # Pass calculated U and V
             packed_output,
             new_base_ptr,
             # --- Dimensions (Passed as constexpr) ---
             N_TOKENS=N_TOKENS, CHANNEL=CHANNEL, N_TOKENS_8=N_TOKENS_8,
             RANK=effective_rank, # Pass the effective rank (1 for mean, K for subspace)
             # --- Strides ---
             stride_xc=x_tensor_cn.stride(0), stride_xn=x_tensor_cn.stride(1),
             stride_bc=base_tensor_cn.stride(0), stride_bn=base_tensor_cn.stride(1),
             stride_scale_uc=scale_u_output_ck.stride(0), stride_scale_uk=scale_u_output_ck.stride(1),
             stride_scale_vk=scale_v_output_kn.stride(0), stride_scale_vn=scale_v_output_kn.stride(1),
             stride_packed_c=packed_output.stride(0), stride_packed_n8=packed_output.stride(1),
             stride_newb_c=stride_newb_c, stride_newb_n=stride_newb_n,
             # --- Meta-parameters (Passed as constexpr) ---
             BLOCK_SIZE_N=BLOCK_SIZE_N,
             UPDATE_CACHE=update_cache,
         )

    # Return values based on update_cache flag
    if update_cache:
        # Return 4 values: packed, u, v, new_base
        return packed_output, scale_u_output_ck, scale_v_output_kn, new_base_output_cn
    else:
        # Always return 4 values, but last one is None if not update_cache
        return packed_output, scale_u_output_ck, scale_v_output_kn, None

# --- Simulation Functions ---

# Helper for manual packing in simulation
def pack_bits_cn_to_cn8(binary_tensor_cn):
    """ Manually packs a C,N boolean/int tensor to C,N//8 uint8 tensor. """
    C, N = binary_tensor_cn.shape
    assert N % 8 == 0
    N_8 = N // 8
    binary_tensor_cn = binary_tensor_cn.to(torch.uint8).contiguous()
    # Reshape to (C, N//8, 8)
    binary_reshaped = binary_tensor_cn.view(C, N_8, 8)
    # Create shifts (0 to 7)
    shifts = torch.arange(0, 8, device=binary_tensor_cn.device, dtype=torch.uint8)
    # Apply shifts and sum (equivalent to bitwise OR)
    packed_cn8 = torch.sum(binary_reshaped << shifts, dim=2, dtype=torch.uint8)
    return packed_cn8

# Helper for manual unpacking in simulation
def unpack_bits_cn8_to_cn(packed_tensor_cn8, N):
    """ Manually unpacks a C,N//8 uint8 tensor to a C,N boolean tensor. """
    C, N_8 = packed_tensor_cn8.shape
    assert N == N_8 * 8
    packed_tensor_cn8 = packed_tensor_cn8.contiguous()
    # Create shifts
    shifts = torch.arange(0, 8, device=packed_tensor_cn8.device, dtype=torch.uint8)
    # Unpack using broadcasting and bitwise AND
    # Expand packed tensor: (C, N//8, 1)
    # Expand shifts: (1, 1, 8)
    # Resulting unpacked shape: (C, N//8, 8)
    unpacked_bits = ((packed_tensor_cn8.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
    # Reshape back to (C, N)
    binary_cn = unpacked_bits.view(C, N)
    return binary_cn


# Simulation uses slowpath quant/dequant OR manual packing/scaling for rank=-1
def sim_binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    rank: int,
    update_cache: bool,
):
    """
    Simulated version of binary_quant_fastpath.
    Uses manual packing/scaling for rank=-1.
    Uses slowpath quantize_1bit for rank > 0.
    Returns: packed(C, N//8), scale_u(C,K or C,1), scale_v(K,N or 1,N), new_base(C,N)|None
    """
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"

    C, N = x_tensor_cn.shape
    new_base_cn = None

    # Calculate tensor to quantize (Always delta)
    tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn

    if rank == -1:
        # --- Simulate Channel-wise Mean Scaling ---
        with Profiler.scope("compact.sim_quant.scale_channel_mean"):
            mean_scale_c1 = torch.mean(torch.abs(tensor_to_quantize_cn), dim=1, keepdim=True)

        # Scales to return, matching fastpath signature for rank=-1
        scale_u_output_ck_sim = mean_scale_c1.contiguous().to(torch.half) # Shape (C, 1)
        scale_v_output_kn_sim = torch.ones((1, N), device=x_tensor_cn.device, dtype=torch.half) # Shape (1, N)

        # Manual Quantization (Packing)
        with Profiler.scope("compact.sim_quant.manual_pack"):
            binary_cn = (tensor_to_quantize_cn >= 0)
            packed_sim = pack_bits_cn_to_cn8(binary_cn) # Manually pack

        if update_cache:
            # Manual Dequantization (for new_base calculation)
            with Profiler.scope("compact.sim_quant.manual_dequant"):
                signs_cn = torch.where(binary_cn, 1, -1).to(torch.int8)
                # Use the calculated mean scale for reconstruction
                recv_quantized_cn = signs_cn * scale_u_output_ck_sim # (C,N) * (C,1) -> (C,N)
                new_base_cn = base_tensor_cn + recv_quantized_cn.to(base_tensor_cn.dtype)

    else: # rank > 0
        # --- Simulate Rank-K Scaling using subspace_iter and quantize_1bit ---
        # Calculate Scales U(C,K), V(K,N) ONCE to match fastpath wrapper output
        with Profiler.scope(f"compact.sim_quant.scale_rank{rank}_approx"):
            scale_U_ck_sim, scale_V_t_kn_sim, _ = subspace_iter(
                torch.abs(tensor_to_quantize_cn), rank=rank, num_iters=2
            )
        # These are the scales returned, matching the fastpath signature
        scale_u_output_ck_sim = scale_U_ck_sim.contiguous().to(torch.half)       # Shape (C, K)
        scale_v_output_kn_sim = scale_V_t_kn_sim.contiguous().to(torch.half) # Shape (K, N)

        # Perform Quantization using slowpath quantize_1bit (ONLY for packed_sim)
        # We ignore the scales returned by slowpath quantize_1bit as we return the ones above
        with Profiler.scope("compact.sim_quant.slowpath_quantize_1bit"):
            tensor_to_quantize_nc = tensor_to_quantize_cn.transpose(0, 1).contiguous()
            # Note: quantize_1bit internally calculates its own scales based on the input tensor
            packed_sim, _, _ = quantize_1bit(tensor_to_quantize_nc.to(torch.float16), rank=rank)

        if update_cache:
            # --- Perform Dequantization using slowpath dequantize_1bit ---
            # Prepare scales for slowpath dequantize_1bit: u_nk, v_kc
            # Use the scales calculated above (identically to the fastpath wrapper)
            # We have U(C,K) and V(K,N). Slowpath needs U(N,K) and V(K,C).
            # u_nk_slowpath = V.T, v_kc_slowpath = U.T
            with Profiler.scope("compact.sim_quant.slowpath_dequantize_1bit"):
                u_nk_for_slowpath = scale_v_output_kn_sim.transpose(0, 1).contiguous() # Shape (N, K)
                v_kc_for_slowpath = scale_u_output_ck_sim.transpose(0, 1).contiguous() # Shape (K, C)

                recv_quantized_nc = dequantize_1bit(packed_sim, u_nk_for_slowpath, v_kc_for_slowpath)
                recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)

            # Calculate new base (Always level 1)
            new_base_cn = base_tensor_cn + recv_quantized_cn

    # Return values matching fastpath signature: packed, scale_u, scale_v, new_base
    return packed_sim, scale_u_output_ck_sim, scale_v_output_kn_sim, new_base_cn


@triton.jit
def _binary_dequant_fastpath(
    # Input Pointers
    packed_in_ptr,     # Packed delta (C, N//8) uint8
    scale_u_ptr,       # Scale factor u (C, K or C, 1)
    scale_v_ptr,       # Scale factor v (K, N or 1, N)
    base_ptr,          # Base cache (C, N) half
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (C, N) half
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    RANK: tl.constexpr,      # Effective rank (1 for mean, K for subspace)
    # Strides
    stride_packed_c, stride_packed_n8,
    stride_scale_uc, stride_scale_uk, # Strides for U (C, K or C, 1)
    stride_scale_vk, stride_scale_vn, # Strides for V (K, N or 1, N)
    stride_base_c, stride_base_n,
    stride_recon_c, stride_recon_n,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
):
    """
    Dequantizes delta and calculates reconstructed activation (base + recv_delta) using rank-K scale calculated INSIDE kernel.
    RANK is the effective rank passed by the wrapper (1 for channel-wise mean, K for subspace).
    Grid: (CHANNEL, cdiv(N_TOKENS, BLOCK_SIZE_N))
    """
    pid_c = tl.program_id(0); pid_n_block = tl.program_id(1)
    n_block_start = pid_n_block * BLOCK_SIZE_N
    offs_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N_TOKENS

    # --- Dequantize Block ---
    # --- Load Scale Components ---
    offs_k = tl.arange(0, RANK) # Loop uses effective rank
    # Load U vector for current channel pid_c: shape (K,) or (1,)
    # Stride uk might be 0 if RANK=1
    scale_u_ptr_base = scale_u_ptr + pid_c * stride_scale_uc
    scale_u_vec = tl.load(scale_u_ptr_base + offs_k * stride_scale_uk) # (RANK,)

    # Load V block for current N block: shape (RANK, BLOCK_SIZE_N)
    offs_n_masked = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    # V layout is (RANK, N), stride_scale_vk = stride K dim, stride_scale_vn = stride N dim
    # Stride vk might be 0 if RANK=1
    offs_v = offs_k[:, None] * stride_scale_vk + offs_n_masked[None, :] * stride_scale_vn
    mask_v = mask_n[None, :] & (offs_k[:, None] < RANK) # Shape (RANK, BLOCK_SIZE_N)
    scale_v_block = tl.load(scale_v_ptr + offs_v, mask=mask_v, other=0.0) # (RANK, BLOCK_SIZE_N)

    # --- Calculate Scale (Rank-K or Rank-1 dot product) ---
    # scale = dot(u_k, v_k) -> sum(u_k * v_k)
    # If RANK=1, this is just u[0] * v[0, :] which is mean_scale * 1.0
    scale_block = tl.sum(scale_u_vec[:, None] * scale_v_block, axis=0).to(tl.float16) # (BLOCK_SIZE_N,)

    # Load and unpack bits
    n8_block_start = n_block_start // 8
    offs_n8 = n8_block_start + tl.arange(0, BLOCK_SIZE_N // 8)
    mask_n8 = offs_n8 < N_TOKENS_8
    packed_row_ptr = packed_in_ptr + pid_c * stride_packed_c
    # Corrected unpacking: Load bytes corresponding to offs_n, then extract bits
    byte_indices_in_row = offs_n // 8
    bit_indices_in_byte = offs_n % 8
    final_byte_mask = mask_n & (byte_indices_in_row < N_TOKENS_8) # Mask for valid bytes within N
    packed_bytes_for_elems = tl.load(packed_row_ptr + byte_indices_in_row * stride_packed_n8, mask=final_byte_mask, other=0)
    bits = ((packed_bytes_for_elems >> bit_indices_in_byte) & 1)

    # --- Calculate Reconstructed Delta ---
    signs = tl.where(mask_n, (2 * bits - 1).to(tl.int8), 0)
    recv_quantized_block = signs * scale_block # This is recv_delta

    # --- Load Base ---
    base_row_ptr = base_ptr + pid_c * stride_base_c
    base_block = tl.load(base_row_ptr + offs_n * stride_base_n, mask=mask_n, other=0.0)

    # --- Calculate Output Block (Always level 1) ---
    recon_block = base_block + recv_quantized_block # recon = base + recv_delta

    # --- Store Output Block ---
    recon_out_ptr = recon_ptr + pid_c * stride_recon_c
    tl.store(recon_out_ptr + offs_n * stride_recon_n, recon_block, mask=mask_n)


@Profiler.prof_func("compact.binary_dequant_fastpath")
def binary_dequant_fastpath(
    packed_in_cn8: torch.Tensor,    # Input packed delta (C, N//8) uint8
    scale_u_ck: torch.Tensor,       # Input scale u (C, K or C, 1)
    scale_v_kn: torch.Tensor,       # Input scale v (K, N or 1, N)
    base_cn: torch.Tensor,         # Input base cache (C, N) half
    # rank: int, # Rank is inferred from scales now
):
    """
    Dequantizes delta and calculates reconstructed activation (base + recv_delta).
    Scale calculation (U @ V) happens INSIDE the kernel.
    Handles both rank-K and rank=-1 (channel-wise mean) cases based on scale shapes.

    Input: packed(C, N//8), u(C,K or C,1), v(K,N or 1,N), base(C,N)
    Output: reconstructed(C, N)
    """
    # Assertions
    # assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1" # Rank no longer explicit input
    assert packed_in_cn8.dtype == torch.uint8
    assert scale_u_ck.dtype == torch.half and scale_v_kn.dtype == torch.half
    assert base_cn.dtype == torch.half
    assert packed_in_cn8.ndim == 2 and scale_u_ck.ndim == 2 and scale_v_kn.ndim == 2 and base_cn.ndim == 2

    assert packed_in_cn8.is_cuda and scale_u_ck.is_cuda and scale_v_kn.is_cuda and base_cn.is_cuda

    packed_in_cn8 = packed_in_cn8.contiguous()
    scale_u_ck = scale_u_ck.contiguous()
    scale_v_kn = scale_v_kn.contiguous()
    base_cn = base_cn.contiguous()

    CHANNEL, N_TOKENS_8 = packed_in_cn8.shape
    N_TOKENS = N_TOKENS_8 * 8
    effective_rank = scale_u_ck.shape[1] # Infer rank from scale U
    assert effective_rank >= 1, "Inferred rank from scale_u must be >= 1"
    assert scale_v_kn.shape[0] == effective_rank, f"Scale V rank mismatch: {scale_v_kn.shape[0]} vs inferred {effective_rank}"

    assert base_cn.shape == (CHANNEL, N_TOKENS), f"Base shape mismatch: {base_cn.shape} vs expected {(CHANNEL, N_TOKENS)}"
    assert scale_u_ck.shape == (CHANNEL, effective_rank), f"Scale U shape mismatch: {scale_u_ck.shape} vs expected {(CHANNEL, effective_rank)}"
    assert scale_v_kn.shape == (effective_rank, N_TOKENS), f"Scale V shape mismatch: {scale_v_kn.shape} vs expected {(effective_rank, N_TOKENS)}"

    # Allocate output tensors
    reconstructed_output_cn = torch.empty_like(base_cn)

    BLOCK_SIZE_N = 512
    assert BLOCK_SIZE_N % 8 == 0, "BLOCK_SIZE_N must be divisible by 8 for unpacking logic"
    grid = (CHANNEL, triton.cdiv(N_TOKENS, BLOCK_SIZE_N))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = base_cn # Use existing tensor for properties

    with Profiler.scope("compact._binary_dequant_fastpath"):
        _binary_dequant_fastpath[grid](
            packed_in_cn8,
            scale_u_ck, scale_v_kn,
            base_cn,
            reconstructed_output_cn,
            # --- Dimensions (Passed as constexpr) ---
            N_TOKENS=N_TOKENS, CHANNEL=CHANNEL, N_TOKENS_8=N_TOKENS_8,
            RANK=effective_rank, # Pass inferred effective rank
            # --- Strides ---
            stride_packed_c=packed_in_cn8.stride(0), stride_packed_n8=packed_in_cn8.stride(1),
            stride_scale_uc=scale_u_ck.stride(0), stride_scale_uk=scale_u_ck.stride(1),
            stride_scale_vk=scale_v_kn.stride(0), stride_scale_vn=scale_v_kn.stride(1),
            stride_base_c=base_cn.stride(0), stride_base_n=base_cn.stride(1),
            stride_recon_c=reconstructed_output_cn.stride(0), stride_recon_n=reconstructed_output_cn.stride(1),
            # --- Meta-parameters (Passed as constexpr) ---
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

    # Return only recon
    return reconstructed_output_cn

# Simulation uses slowpath dequant OR manual dequant for rank=-1
def sim_binary_dequant_fastpath(
    packed_in_cn8: torch.Tensor, # Packed bits (C, N//8) UINT8
    scale_u_ck: torch.Tensor,     # Scale factor U (C, K or C, 1)
    scale_v_kn: torch.Tensor,     # Scale factor V (K, N or 1, N)
    base_cn: torch.Tensor,
    # rank: int, # Rank is inferred
):
    """
    Simulated version of binary_dequant_fastpath.
    Uses the general slowpath dequantize_1bit function based on the provided scales.
    Handles both rank-K and rank-1 cases correctly.
    Returns: reconstructed(C,N)
    """
    # Infer rank from scales
    effective_rank = scale_u_ck.shape[1]
    assert effective_rank >= 1, "Inferred rank must be >= 1"
    assert scale_v_kn.shape[0] == effective_rank, "Scale V rank mismatch"

    C = scale_u_ck.shape[0]
    N = scale_v_kn.shape[1]
    assert packed_in_cn8.shape == (C, N // 8)
    assert scale_u_ck.shape == (C, effective_rank)
    assert scale_v_kn.shape == (effective_rank, N)

    # --- Always use the slowpath dequantize_1bit for simulation consistency --- 
    # It handles rank=1 correctly when provided with appropriate rank-1 U/V scales.
    with Profiler.scope("compact.sim_dequant.slowpath_dequantize_1bit"):
        # slowpath expects u(N,K) and v(K,C).
        # We have u(C,K) and v(K,N) from fastpath.
        # Map: u_nk_slowpath = scale_v_kn.T, v_kc_slowpath = scale_u_ck.T
        u_nk_for_slowpath = scale_v_kn.transpose(0, 1).contiguous() # Shape (N, K)
        v_kc_for_slowpath = scale_u_ck.transpose(0, 1).contiguous() # Shape (K, C)

        recv_quantized_nc = dequantize_1bit(packed_in_cn8, u_nk_for_slowpath, v_kc_for_slowpath)
        recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)
        reconstructed_cn = base_cn + recv_quantized_cn

    # Return only reconstructed
    return reconstructed_cn


# --- Test Functions ---
def profile_quantize_kernels(num_runs=1000, num_warmup=5):
    """Profile quantize kernel."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    atol, rtol = 1e-3, 1e-2
    # Test both rank=4 and rank=-1
    RANKS_TO_TEST = [4, -1] # <<< Set Ranks for testing

    print("--- Quant Performance & Correctness --- (ms per run)")

    for rank_test in RANKS_TO_TEST:
        print(f"\\nTesting Rank: {rank_test}")
        # Setup Tensors
        x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
        base_tensor_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()

        # Sim args (simplified)
        sim_args = (x_tensor_cn, base_tensor_cn, rank_test, True) # update_cache = True

        # --- Calculate Reference Values (Simulation) ---
        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            ref_packed, ref_scale_u, ref_scale_v, ref_new_base = sim_binary_quant_fastpath(*sim_args)

        # --- Warm-up runs ---
        for _ in range(num_warmup):
            with torch.random.fork_rng(devices=['cuda']):
                 torch.manual_seed(42)
                 _ = binary_quant_fastpath(
                     x_tensor_cn, base_tensor_cn,
                     rank=rank_test,
                     update_cache=True,
                 )
                 _ = sim_binary_quant_fastpath(*sim_args)
            torch.cuda.synchronize()

        # --- Get Kernel results for correctness check ---
        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            kernel_packed, kernel_scale_u, kernel_scale_v, kernel_new_base = binary_quant_fastpath(
                x_tensor_cn, base_tensor_cn,
                rank=rank_test,
                update_cache=True,
            )
        torch.cuda.synchronize()

        # --- Profiling Kernel ---
        torch.cuda.synchronize()
        start_kernel = time.time()
        for i in range(num_runs):
            # Alternate update_cache to test both paths if applicable
            update_c = (i % 2 == 0)
            with torch.random.fork_rng(devices=['cuda']):
                 torch.manual_seed(42 + i) # Vary seed slightly
                 _ = binary_quant_fastpath(
                     x_tensor_cn, base_tensor_cn,
                     rank=rank_test,
                     update_cache=update_c,
                 )
        torch.cuda.synchronize()
        end_kernel = time.time()
        kernel_time = (end_kernel - start_kernel) / num_runs
        print(f"Kernel (Rank {rank_test}): {kernel_time*1000:.3f} ms")

        # --- Verify correctness (vs Sim with update_cache=True) ---
        correct = True
        issues = []
        # Packed
        if not torch.equal(ref_packed, kernel_packed):
            correct = False; issues.append(f"Packed (Max Diff: {torch.max(torch.abs(ref_packed.float() - kernel_packed.float()))})")
        # Scale U
        if not torch.allclose(ref_scale_u, kernel_scale_u, atol=atol, rtol=rtol):
            correct = False; issues.append(f"Scale U (Max Diff: {torch.max(torch.abs(ref_scale_u - kernel_scale_u))})")
        # Scale V
        if not torch.allclose(ref_scale_v, kernel_scale_v, atol=atol, rtol=rtol):
             correct = False; issues.append(f"Scale V (Max Diff: {torch.max(torch.abs(ref_scale_v - kernel_scale_v))})")
        # New Base
        if ref_new_base is not None and kernel_new_base is not None:
             # Increase tolerance slightly for fp16 accumulation differences
             if not torch.allclose(ref_new_base, kernel_new_base, atol=atol*5, rtol=rtol*5):
                 max_diff_base = torch.max(torch.abs(ref_new_base - kernel_new_base))
                 correct = False; issues.append(f"New Base (Max Diff: {max_diff_base})")
        elif ref_new_base is not None or kernel_new_base is not None:
             correct = False; issues.append("New Base (Mismatch None/Not None)")

        print(f"Correctness vs Sim (Rank {rank_test}, Update Cache=True): {correct}")
        if not correct: print(f"  Issues: {', '.join(issues)}\\n")

    print("---")


def profile_dequantize_kernels(num_runs=1000, num_warmup=5):
    """Profile dequantize kernel."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    atol, rtol = 1e-3, 1e-2
    # Test both rank=4 and rank=-1
    RANKS_TO_TEST = [4, -1] # <<< Set Ranks for testing

    print("--- Dequant Performance & Correctness --- (ms per run)")

    for rank_test in RANKS_TO_TEST:
        print(f"\\nTesting Rank: {rank_test}")
        # Setup Tensors
        x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
        base_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()

        # --- Generate Inputs using Quant Sim (no cache update needed here) ---
        quant_sim_args = (x_tensor_cn, base_cn, rank_test, False) # update_cache = False

        with torch.random.fork_rng(devices=['cuda']):
            torch.manual_seed(42)
            # Quant sim returns 4 values now, we need packed and scales
            input_packed, input_scale_u, input_scale_v, _ = sim_binary_quant_fastpath(*quant_sim_args)

        # --- Calculate Reference Outputs (Dequant Simulation) ---
        # Dequant sim args
        dequant_sim_args = (input_packed, input_scale_u, input_scale_v, base_cn)

        with torch.random.fork_rng(devices=['cuda']):
             torch.manual_seed(43) # Use different seed just in case
             ref_reconstructed_cn = sim_binary_dequant_fastpath(*dequant_sim_args)

        # --- Warm-up runs ---
        for _ in range(num_warmup):
             with torch.random.fork_rng(devices=['cuda']):
                 torch.manual_seed(43)
                 _ = binary_dequant_fastpath(
                     input_packed, input_scale_u, input_scale_v, base_cn
                 )
                 _ = sim_binary_dequant_fastpath(*dequant_sim_args)
             torch.cuda.synchronize()

        # --- Get Kernel results for correctness check ---
        with torch.random.fork_rng(devices=['cuda']):
             torch.manual_seed(43)
             kernel_reconstructed_cn = binary_dequant_fastpath(
                 input_packed, input_scale_u, input_scale_v, base_cn
             )
        torch.cuda.synchronize()

        # --- Profiling Kernel ---
        torch.cuda.synchronize()
        start_kernel = time.time()
        for _ in range(num_runs):
            _ = binary_dequant_fastpath(
                input_packed, input_scale_u, input_scale_v, base_cn
            )
        torch.cuda.synchronize()
        end_kernel = time.time()
        kernel_time = (end_kernel - start_kernel) / num_runs
        print(f"Kernel (Rank {rank_test}): {kernel_time*1000:.3f} ms")

        # --- Verify correctness ---
        correct = True
        issues = []
        # Reconstructed
        # Increase tolerance slightly for fp16 accumulation differences
        if not torch.allclose(ref_reconstructed_cn, kernel_reconstructed_cn, atol=atol*5, rtol=rtol*5):
            max_diff_recon = torch.max(torch.abs(ref_reconstructed_cn - kernel_reconstructed_cn))
            correct = False; issues.append(f"Reconstructed (Max Diff: {max_diff_recon})")

        print(f"Correctness vs Sim (Rank {rank_test}): {correct}")
        if not correct: print(f"  Issues: {', '.join(issues)}\\n")

    print("---")


if __name__ == "__main__":
    # Add helpers to global scope if needed, or keep them local
    # Test functions only need to run once now
    profile_quantize_kernels()
    profile_dequantize_kernels()
