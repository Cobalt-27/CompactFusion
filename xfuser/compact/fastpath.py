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
    scale_u_ptr,       # Input scale factor u (C, K)
    scale_v_ptr,       # Input scale factor v (K, N)
    # Output Pointers
    packed_out_ptr,    # Packed delta (C, N//8) - OUTPUT
    new_base_ptr,      # Output new base (C, N) - optional write
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    RANK: tl.constexpr,      # Rank K for scale approximation
    # Strides (ALL are for C, N layout, except packed and scale)
    stride_xc, stride_xn,
    stride_bc, stride_bn,
    stride_scale_uc, stride_scale_uk, # Stride for u (C, K)
    stride_scale_vn, stride_scale_vk, # Stride for v (K, N)
    stride_packed_c, stride_packed_n8, # Strides for packed output (C, N//8)
    stride_newb_c, stride_newb_n,      # Strides for new_base (C, N)
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
    UPDATE_CACHE: tl.constexpr,
):
    """
    Quantizes delta (x - base) to 1-bit.
    Packs into 1-bit representation using sign. Scale is calculated using rank-K approximation INSIDE kernel.
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
        # --- Load Scale Components (Rank-K) ---
        offs_k = tl.arange(0, RANK)
        # Load U vector for current channel pid_c: shape (K,)
        scale_u_ptr_base = scale_u_ptr + pid_c * stride_scale_uc
        scale_u_vec = tl.load(scale_u_ptr_base + offs_k * stride_scale_uk) # (K,)

        # Load V block for current N block: shape (K, BLOCK_SIZE_N) from V (K, N)
        offs_n_masked = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        # Correct broadcasting for V offsets: (K, 1) for k_offsets, (1, BLOCK_SIZE_N) for n_offsets
        # V layout is (K, N), stride_scale_vk = stride K dim, stride_scale_vn = stride N dim
        offs_v = offs_k[:, None] * stride_scale_vk + offs_n_masked[None, :] * stride_scale_vn # <<< CHANGED Calculation
        mask_v = mask_n[None, :] & (offs_k[:, None] < RANK) # Shape (K, BLOCK_SIZE_N)
        scale_v_block = tl.load(scale_v_ptr + offs_v, mask=mask_v, other=0.0) # (K, BLOCK_SIZE_N)

        # --- Calculate Scale (Rank-K dot product) ---
        # scale = dot(u_k, v_k) -> sum(u_k * v_k)
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
    rank: int,                        # Rank K
    update_cache: bool,
):
    """
    Quantizes delta (x - base) to 1-bit using fast path kernel.
    Calculates Rank-K scale factors U(C,K), V(K,N) and passes them to kernel for internal scale calculation.
    Returns: packed(C, N//8), scale_u(C,K), scale_v(K,N), new_base(C,N)|None
    """
    # Assertions
    assert rank >= 1, "Rank must be >= 1"
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

    # Calculate rank-K approximation for scale based on abs(tensor_to_quantize)
    # subspace_iter expects (C, N), returns U_ck(C, K), V_t_kn(K, N)
    with Profiler.scope(f"compact.quant.scale_rank{rank}_approx"):
        scale_U_ck, scale_V_t_kn, _ = subspace_iter(
            torch.abs(tensor_to_quantize_cn), rank=rank, num_iters=2
        )
    # Kernel expects U(C, K) and V(K, N)
    scale_u_output_ck = scale_U_ck.contiguous().to(torch.half)       # Shape (C, K)
    scale_v_output_kn = scale_V_t_kn.contiguous().to(torch.half) # Shape (K, N)
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
             scale_u_output_ck, scale_v_output_kn,
             packed_output,
             new_base_ptr,
             # --- Dimensions (Passed as constexpr) ---
             N_TOKENS=N_TOKENS, CHANNEL=CHANNEL, N_TOKENS_8=N_TOKENS_8,
             RANK=rank,
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

# Simulation uses slowpath quant/dequant
def sim_binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    rank: int,
    update_cache: bool,
):
    """
    Simulated version of binary_quant_fastpath using slowpath quantize_1bit and dequantize_1bit.
    Handles only RESIDUAL LEVEL 1.
    Calculates scales U(C,K), V(K,N) exactly once to match fastpath wrapper's calculation and return signature.
    Uses these calculated scales (transposed) for the internal dequantization simulation if update_cache=True.
    Returns: packed(C, N//8), scale_u(C,K), scale_v(K,N), new_base(C,N)|None
    """
    assert rank >= 1, "Rank must be >= 1"

    C, N = x_tensor_cn.shape
    new_base_cn = None

    # Calculate tensor to quantize (Always delta)
    tensor_to_quantize_cn = x_tensor_cn - base_tensor_cn

    # --- Calculate Scales U(C,K), V(K,N) ONCE --- Mimics fastpath wrapper
    with Profiler.scope(f"compact.sim_quant.scale_rank{rank}_approx"):
        scale_U_ck_sim, scale_V_t_kn_sim, _ = subspace_iter(
            torch.abs(tensor_to_quantize_cn), rank=rank, num_iters=2
        )
    # These are the scales returned, matching the fastpath signature
    scale_u_output_ck_sim = scale_U_ck_sim.contiguous().to(torch.half)       # Shape (C, K)
    scale_v_output_kn_sim = scale_V_t_kn_sim.contiguous().to(torch.half) # Shape (K, N)

    # --- Perform Quantization using slowpath quantize_1bit (ONLY for packed_sim) ---
    # We ignore the scales returned by slowpath quantize_1bit
    tensor_to_quantize_nc = tensor_to_quantize_cn.transpose(0, 1).contiguous()
    packed_sim, _, _ = quantize_1bit(tensor_to_quantize_nc.to(torch.float16), rank=rank)

    if update_cache:
        # --- Perform Dequantization using slowpath dequantize_1bit ---
        # Prepare scales for slowpath dequantize_1bit: u_nk, v_kc
        # Use the scales calculated above (identically to the fastpath wrapper)
        # We have U(C,K) and V(K,N). Slowpath needs U(N,K) and V(K,C).
        # u_nk_slowpath = V.T, v_kc_slowpath = U.T
        u_nk_for_slowpath = scale_v_output_kn_sim.transpose(0, 1).contiguous() # Shape (N, K)
        v_kc_for_slowpath = scale_u_output_ck_sim.transpose(0, 1).contiguous() # Shape (K, C)

        recv_quantized_nc = dequantize_1bit(packed_sim, u_nk_for_slowpath, v_kc_for_slowpath)
        recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)

        # Calculate new base (Always level 1)
        new_base_cn = base_tensor_cn + recv_quantized_cn

    # Return values matching fastpath signature: packed, scale_u(C,K), scale_v(K,N), new_base
    return packed_sim, scale_u_output_ck_sim, scale_v_output_kn_sim, new_base_cn


@triton.jit
def _binary_dequant_fastpath(
    # Input Pointers
    packed_in_ptr,     # Packed delta (C, N//8) uint8
    scale_u_ptr,       # Scale factor u (C, K)
    scale_v_ptr,       # Scale factor v (K, N)
    base_ptr,          # Base cache (C, N) half
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (C, N) half
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    RANK: tl.constexpr,
    # Strides
    stride_packed_c, stride_packed_n8,
    stride_scale_uc, stride_scale_uk,
    stride_scale_vk, stride_scale_vn,
    stride_base_c, stride_base_n,
    stride_recon_c, stride_recon_n,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, # Block size for N dimension
):
    """
    Dequantizes delta and calculates reconstructed activation (base + recv_delta) using rank-K scale calculated INSIDE kernel.
    Grid: (CHANNEL, cdiv(N_TOKENS, BLOCK_SIZE_N))
    """
    pid_c = tl.program_id(0); pid_n_block = tl.program_id(1)
    n_block_start = pid_n_block * BLOCK_SIZE_N
    offs_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N_TOKENS

    # --- Dequantize Block ---
    # --- Load Scale Components (Rank-K) ---
    offs_k = tl.arange(0, RANK)
    # Load U vector for current channel pid_c: shape (K,)
    scale_u_ptr_base = scale_u_ptr + pid_c * stride_scale_uc
    scale_u_vec = tl.load(scale_u_ptr_base + offs_k * stride_scale_uk) # (K,)

    # Load V block for current N block: shape (K, BLOCK_SIZE_N) from V (K, N)
    offs_n_masked = n_block_start + tl.arange(0, BLOCK_SIZE_N)
    # Correct broadcasting for V offsets: (K, 1) for k_offsets, (1, BLOCK_SIZE_N) for n_offsets
    # V layout is (K, N), stride_scale_vk = stride K dim, stride_scale_vn = stride N dim
    offs_v = offs_k[:, None] * stride_scale_vk + offs_n_masked[None, :] * stride_scale_vn # <<< CHANGED Calculation
    mask_v = mask_n[None, :] & (offs_k[:, None] < RANK) # Shape (K, BLOCK_SIZE_N)
    scale_v_block = tl.load(scale_v_ptr + offs_v, mask=mask_v, other=0.0) # (K, BLOCK_SIZE_N)

    # --- Calculate Scale (Rank-K dot product) ---
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
    scale_u_ck: torch.Tensor,       # Input scale u (C, K)
    scale_v_kn: torch.Tensor,       # Input scale v (K, N)
    base_cn: torch.Tensor,         # Input base cache (C, N) half
    rank: int,
):
    """
    Dequantizes delta and calculates reconstructed activation (base + recv_delta) using rank-K scales.
    Scale calculation (U @ V) happens INSIDE the kernel.

    Input: packed(C, N//8), u(C,K), v(K,N), base(C,N), rank
    Output: reconstructed(C, N)
    """
    # Assertions
    assert rank >= 1, "Rank must be >= 1"
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
    assert base_cn.shape == (CHANNEL, N_TOKENS), f"Base shape mismatch: {base_cn.shape} vs expected {(CHANNEL, N_TOKENS)}"
    assert scale_u_ck.shape == (CHANNEL, rank), f"Scale U shape mismatch: {scale_u_ck.shape} vs expected {(CHANNEL, rank)}"
    assert scale_v_kn.shape == (rank, N_TOKENS), f"Scale V shape mismatch: {scale_v_kn.shape} vs expected {(rank, N_TOKENS)}"

    # Allocate output tensors
    reconstructed_output_cn = torch.empty_like(base_cn)

    BLOCK_SIZE_N = 512
    assert BLOCK_SIZE_N % 8 == 0, "BLOCK_SIZE_N must be divisible by 8 for unpacking logic"
    grid = (CHANNEL, triton.cdiv(N_TOKENS, BLOCK_SIZE_N))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = base_cn # Use existing tensor for properties

    # with Profiler.scope("compact._binary_dequant_fastpath"):
    _binary_dequant_fastpath[grid](
        packed_in_cn8,
        scale_u_ck, scale_v_kn,
        base_cn,
        reconstructed_output_cn,
        # --- Dimensions (Passed as constexpr) ---
        N_TOKENS=N_TOKENS, CHANNEL=CHANNEL, N_TOKENS_8=N_TOKENS_8,
        RANK=rank,
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

# Simulation uses slowpath quant/dequant
def sim_binary_dequant_fastpath(
    packed_in_cn8: torch.Tensor, # Packed bits (C, N//8) UINT8
    scale_u_ck: torch.Tensor,     # Scale factor U (C, K)
    scale_v_kn: torch.Tensor,     # Scale factor V (K, N)
    base_cn: torch.Tensor,
    rank: int,
):
    """
    Simulated version of binary_dequant_fastpath using slowpath dequantize_1bit.
    Accepts scales u(C,K), v(K,N) consistent with kernel path signature.
    Transforms scales to u(N,K), v(K,C) expected by slowpath dequantize_1bit.
    Handles only RESIDUAL LEVEL 1.
    Returns: reconstructed(C,N)
    """
    assert rank >= 1, "Rank must be >= 1"

    C = scale_u_ck.shape[0]
    N = scale_v_kn.shape[1]
    assert packed_in_cn8.shape == (C, N // 8)
    assert scale_u_ck.shape == (C, rank)
    assert scale_v_kn.shape == (rank, N)

    # --- Perform Dequantization using slowpath dequantize_1bit ---
    # slowpath expects u(N,K) and v(K,C).
    # We have u(C,K) and v(K,N) from fastpath.
    # Map: u_nk_slowpath = scale_v_kn.T, v_kc_slowpath = scale_u_ck.T
    u_nk_for_slowpath = scale_v_kn.transpose(0, 1).contiguous() # Shape (N, K)
    v_kc_for_slowpath = scale_u_ck.transpose(0, 1).contiguous() # Shape (K, C)

    recv_quantized_nc = dequantize_1bit(packed_in_cn8, u_nk_for_slowpath, v_kc_for_slowpath)
    recv_quantized_cn = recv_quantized_nc.transpose(0, 1).contiguous().to(torch.float16)

    reconstructed_cn = None

    # Calculate reconstructed activation (Level 1)
    reconstructed_cn = base_cn + recv_quantized_cn

    # Return only reconstructed
    return reconstructed_cn


# --- Test Functions ---
def profile_quantize_kernels(num_runs=1000, num_warmup=5):
    """Profile quantize kernel."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    atol, rtol = 1e-3, 1e-2
    RANK_TO_TEST = 4 # <<< Set Rank for testing

    print("--- Quant Performance & Correctness --- (ms per run)")

    print(f"\nTesting Rank: {RANK_TO_TEST}")
    # Setup Tensors
    x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
    base_tensor_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()

    # Sim args (simplified)
    sim_args = (x_tensor_cn, base_tensor_cn, RANK_TO_TEST, True)

    # --- Calculate Reference Values (Simulation) ---
    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(42)
        # Sim function returns 4 values now
        ref_packed, ref_scale_u, ref_scale_v_kn, ref_new_base = sim_binary_quant_fastpath(*sim_args)

    # --- Warm-up runs ---
    for _ in range(num_warmup):
        with torch.random.fork_rng(devices=['cuda']):
             torch.manual_seed(42)
             _ = binary_quant_fastpath(
                 x_tensor_cn, base_tensor_cn,
                 rank=RANK_TO_TEST,
                 update_cache=True,
             )
             _ = sim_binary_quant_fastpath(*sim_args)
        torch.cuda.synchronize()

    # --- Get Kernel results for correctness check ---
    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(42)
        # Kernel returns 4 values now
        kernel_packed, kernel_scale_u, kernel_scale_v_kn, kernel_new_base = binary_quant_fastpath(
            x_tensor_cn, base_tensor_cn,
            rank=RANK_TO_TEST,
            update_cache=True,
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
                 x_tensor_cn, base_tensor_cn,
                 rank=RANK_TO_TEST,
                 update_cache=update_c,
             )
    torch.cuda.synchronize()
    end_kernel = time.time()
    kernel_time = (end_kernel - start_kernel) / num_runs
    print(f"Kernel (Rank {RANK_TO_TEST}): {kernel_time*1000:.3f} ms")

    # --- Verify correctness ---
    correct = True
    issues = []
    # Packed
    if not torch.equal(ref_packed, kernel_packed):
        correct = False; issues.append("Packed")
    # Scale U
    if not torch.allclose(ref_scale_u, kernel_scale_u, atol=atol, rtol=rtol):
        correct = False; issues.append(f"Scale U (Max Diff: {torch.max(torch.abs(ref_scale_u - kernel_scale_u))})")
    # Scale V (K,N)
    if not torch.allclose(ref_scale_v_kn, kernel_scale_v_kn, atol=atol, rtol=rtol):
         correct = False; issues.append(f"Scale V (Max Diff: {torch.max(torch.abs(ref_scale_v_kn - kernel_scale_v_kn))})")
    # New Base
    if ref_new_base is not None and kernel_new_base is not None:
         if not torch.allclose(ref_new_base, kernel_new_base, atol=atol, rtol=rtol):
             correct = False; issues.append(f"New Base (Max Diff: {torch.max(torch.abs(ref_new_base - kernel_new_base))})")
    elif ref_new_base is not None or kernel_new_base is not None:
         correct = False; issues.append("New Base (Mismatch None)")

    print(f"Correctness vs Sim (Rank {RANK_TO_TEST}): {correct}")
    if not correct: print(f"  Issues: {', '.join(issues)}\n")
    print("---")


def profile_dequantize_kernels(num_runs=1000, num_warmup=5):
    """Profile dequantize kernel."""
    import time
    N_TOKENS, CHANNEL = 4096, 2048
    atol, rtol = 1e-3, 1e-2
    RANK_TO_TEST = 4 # <<< Set Rank for testing

    print("--- Dequant Performance & Correctness --- (ms per run)")

    print(f"\nTesting Rank: {RANK_TO_TEST}")
    # Setup Tensors
    x_tensor_cn = (torch.randn((CHANNEL, N_TOKENS), dtype=torch.half, device="cuda") * 0.5).contiguous()
    base_cn = (torch.randn_like(x_tensor_cn) * 0.1).contiguous()

    # --- Generate Inputs using Quant Sim ---
    # Quant sim args simplified
    quant_sim_args = (x_tensor_cn, base_cn, RANK_TO_TEST, False)

    with torch.random.fork_rng(devices=['cuda']):
        torch.manual_seed(42)
        # Quant sim returns 4 values now
        input_packed, input_scale_u_ck, input_scale_v_kn, _ = sim_binary_quant_fastpath(*quant_sim_args)

    # --- Calculate Reference Outputs (Dequant Simulation) ---
    # Dequant sim args simplified
    dequant_sim_args = (input_packed, input_scale_u_ck, input_scale_v_kn, base_cn, RANK_TO_TEST)

    # Sim function returns 1 value now
    ref_reconstructed_cn = sim_binary_dequant_fastpath(*dequant_sim_args)

    # --- Warm-up runs ---
    for _ in range(num_warmup):
        _ = binary_dequant_fastpath(
            input_packed, input_scale_u_ck, input_scale_v_kn,
            base_cn,
            rank=RANK_TO_TEST,
        )
        _ = sim_binary_dequant_fastpath(*dequant_sim_args)
        torch.cuda.synchronize()

    # --- Get Kernel results for correctness check ---
    # Kernel returns 1 value now
    kernel_reconstructed_cn = binary_dequant_fastpath(
        input_packed, input_scale_u_ck, input_scale_v_kn,
        base_cn,
        rank=RANK_TO_TEST,
    )
    torch.cuda.synchronize()

    # --- Profiling Kernel ---
    torch.cuda.synchronize()
    start_kernel = time.time()
    for _ in range(num_runs):
        _ = binary_dequant_fastpath(
            input_packed, input_scale_u_ck, input_scale_v_kn,
            base_cn,
            rank=RANK_TO_TEST,
        )
    torch.cuda.synchronize()
    end_kernel = time.time()
    kernel_time = (end_kernel - start_kernel) / num_runs
    print(f"Kernel (Rank {RANK_TO_TEST}): {kernel_time*1000:.3f} ms")

    # --- Verify correctness ---
    correct = True
    issues = []
    # Reconstructed
    if not torch.allclose(ref_reconstructed_cn, kernel_reconstructed_cn, atol=atol, rtol=rtol):
        correct = False; issues.append(f"Reconstructed (Max Diff: {torch.max(torch.abs(ref_reconstructed_cn - kernel_reconstructed_cn))})")

    print(f"Correctness vs Sim (Rank {RANK_TO_TEST}): {correct}")
    if not correct: print(f"  Issues: {', '.join(issues)}\n")
    print("---")


if __name__ == "__main__":
    # Test functions only need to run once now
    profile_quantize_kernels()
    profile_dequantize_kernels()
