import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler
from xfuser.compact.compress_quantize import (
    dequantize_1bit, quantize_1bit,
    quantize_int2, dequantize_int2 # Add imports for INT2 slowpath
)
# Import subspace iter for rank-1 scale approximation
from xfuser.compact.compress_lowrank import subspace_iter

@triton.jit
def _binary_quant_fastpath(
    # Input Pointers (Layout N, C)
    x_ptr,             # Current activation (N, C)
    base_ptr,          # Cached base (N, C)
    scale_u_ptr,       # Input scale factor u (N, K)
    scale_v_ptr,       # Input scale factor v (C, K)
    # Output Pointers
    packed_out_ptr,    # Packed delta (N, C//8) - OUTPUT
    new_base_ptr,      # Output new base (N, C) - optional write
    # Dimensions
    N_ROWS: tl.constexpr,    # Original N
    C_COLS: tl.constexpr,    # Original C
    C_COLS_8: tl.constexpr,  # C_COLS // 8
    RANK: tl.constexpr,       # Effective rank K
    # Strides (ALL are for N, C layout, except packed and scales)
    stride_xn, stride_xc,
    stride_bn, stride_bc,
    stride_scale_un, stride_scale_uk, # Strides for U (N, K)
    stride_scale_vc, stride_scale_vk, # Strides for V (C, K)
    stride_packed_n, stride_packed_c8, # Strides for packed output (N, C//8)
    stride_newb_n, stride_newb_c,      # Strides for new_base (N, C)
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr, # Block size for C dimension
    UPDATE_CACHE: tl.constexpr,
):
    """
    Quantizes delta (x - base) to 1-bit. Input/Output layout (N, C).
    Packs into 1-bit representation along C dim. Output packed layout (N, C//8).
    Scale factors U(N, K), V(C, K) are provided.
    Scale is calculated as dot(U[n,:], V[c,:]) INSIDE kernel if UPDATE_CACHE.
    Optionally updates base cache.
    Grid: (N_ROWS, cdiv(C_COLS, BLOCK_SIZE_C))
    """
    pid_n = tl.program_id(0); pid_c_block = tl.program_id(1)
    c_block_start = pid_c_block * BLOCK_SIZE_C
    offs_c = c_block_start + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < C_COLS

    # --- Load Inputs (Row pid_n, Block of C) --- 
    x_row_ptr = x_ptr + pid_n * stride_xn
    base_row_ptr = base_ptr + pid_n * stride_bn
    x_block = tl.load(x_row_ptr + offs_c * stride_xc, mask=mask_c, other=0.0) # (BLOCK_SIZE_C,)
    base_block = tl.load(base_row_ptr + offs_c * stride_bc, mask=mask_c, other=0.0) # (BLOCK_SIZE_C,)

    # --- Calculate Tensor to Quantize (Always delta) ---
    tensor_to_quantize = x_block - base_block # (BLOCK_SIZE_C,)

    # --- Quantize (Pack Signs along C) --- 
    # Get signs (1 if >=0 else 0) for the block
    binary = (tensor_to_quantize >= 0).to(tl.uint8) # (BLOCK_SIZE_C,)
    
    # Reshape for packing (BLOCK_SIZE_C must be multiple of 8)
    binary_reshaped = tl.reshape(binary, (BLOCK_SIZE_C // 8, 8))
    
    # Create shifts [0, 1, ..., 7]
    shifts = tl.arange(0, 8).to(tl.uint8)
    
    # Shift and sum (equivalent to bitwise OR for non-overlapping bits) to pack 8 bits into uint8
    shifted = (binary_reshaped << shifts).to(tl.uint8) # (BLOCK_SIZE_C//8, 8)
    packed_block = tl.sum(shifted, axis=1).to(tl.uint8) # (BLOCK_SIZE_C//8,)

    # --- Store Packed Block --- 
    # Calculate offsets in the packed dimension C//8
    c8_block_start = c_block_start // 8
    offs_c8 = c8_block_start + tl.arange(0, BLOCK_SIZE_C // 8)
    mask_c8 = offs_c8 < C_COLS_8
    
    # Get base pointer for the output row
    packed_output_row_ptr = packed_out_ptr + pid_n * stride_packed_n
    # Calculate full pointer offsets for the packed block
    packed_output_ptrs = packed_output_row_ptr + offs_c8 * stride_packed_c8
    # Store the packed result
    tl.store(packed_output_ptrs, packed_block, mask=mask_c8)

    # --- Update Cache (Conditional) --- 
    if UPDATE_CACHE:
        # --- Load Scale Components U(N, K), V(C, K) --- 
        offs_k = tl.arange(0, RANK)
        
        # Load U vector for current row pid_n: shape (K,)
        scale_u_row_ptr = scale_u_ptr + pid_n * stride_scale_un
        scale_u_vec = tl.load(scale_u_row_ptr + offs_k * stride_scale_uk) # (RANK,)

        # Load V block for current C block: shape (BLOCK_SIZE_C, RANK)
        # Pointer starts at C block start
        v_block_start_ptr = scale_v_ptr + c_block_start * stride_scale_vc
        # Offsets: (BLOCK_SIZE_C, 1) * stride_vc + (1, RANK) * stride_vk
        offs_v = tl.arange(0, BLOCK_SIZE_C)[:, None] * stride_scale_vc + offs_k[None, :] * stride_scale_vk
        # Mask for V load: (BLOCK_SIZE_C, 1) & (1, RANK)
        mask_v = mask_c[:, None] & (offs_k[None, :] < RANK)
        scale_v_block = tl.load(v_block_start_ptr + offs_v, mask=mask_v, other=0.0) # (BLOCK_SIZE_C, RANK)

        # --- Calculate Scale: dot(U[n,:], V[c,:]) -> sum(U[n,k] * V[c,k]) --- 
        # We have U_vec (RANK,) and V_block (BLOCK_SIZE_C, RANK)
        # Need element-wise product and sum over K
        # scale_block[c] = sum_k ( scale_u_vec[k] * scale_v_block[c, k] )
        scale_block = tl.sum(scale_u_vec[None, :] * scale_v_block, axis=1).to(tl.float16) # (BLOCK_SIZE_C,)

        # Dequantize based on the `binary` block (0/1 values)
        sign_int8 = tl.where(mask_c, (2 * binary.to(tl.int8) - 1), 0) # (BLOCK_SIZE_C,)
        recv_quantized_block = sign_int8 * scale_block # This is recv_delta

        # Calculate new base
        new_base_block = base_block + recv_quantized_block # new_base = base + recv_delta

        # Store new base (Row pid_n, Block of C)
        new_base_row_ptr = new_base_ptr + pid_n * stride_newb_n
        tl.store(new_base_row_ptr + offs_c * stride_newb_c, new_base_block, mask=mask_c)


@Profiler.prof_func("compact.binary_quant_fastpath")
def binary_quant_fastpath(
    x_tensor_nc: torch.Tensor,        # Input (N, C)
    base_tensor_nc: torch.Tensor,     # Input (N, C)
    rank: int,                        # Rank K or -1 for channel-wise mean
    update_cache: bool,
):
    """
    Quantizes delta (x - base) to 1-bit using fast path kernel. Input/Output (N, C).
    Calculates Rank-K scale factors U(N,K), V(C,K) OR channel-wise mean scale (if rank=-1).
    Passes scales U(N, K), V(C, K) to kernel.
    Returns: packed(N, C//8), scale_u(N,K or N,1), scale_v(C,K or C,1), new_base(N,C)|None
    """
    # Assertions
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"
    assert x_tensor_nc.dtype == torch.half
    assert base_tensor_nc.dtype == torch.half
    assert x_tensor_nc.ndim == 2 and base_tensor_nc.ndim == 2
    assert x_tensor_nc.shape == base_tensor_nc.shape

    x_tensor_nc = x_tensor_nc.contiguous()
    base_tensor_nc = base_tensor_nc.contiguous()

    N_ROWS, C_COLS = x_tensor_nc.shape
    assert C_COLS % 8 == 0, "C_COLS must be divisible by 8 for packing output alignment"
    C_COLS_8 = C_COLS // 8

    # Calculate tensor to quantize (Always delta)
    tensor_to_quantize_nc = x_tensor_nc - base_tensor_nc

    # --- Scale Calculation (U(N, K), V(C, K)) --- 
    tensor_to_quantize_abs_nc = torch.abs(tensor_to_quantize_nc)

    if rank == -1:
        # Calculate channel-wise mean scale
        with Profiler.scope("compact.quant.calc_scale_mean"):
            # Scale is the mean of absolute values per channel -> (C,)
            mean_scale_c = torch.mean(tensor_to_quantize_abs_nc, dim=0)
            # Prepare rank-1 structure for the kernel: U(N, 1) = ones, V(C, 1) = mean_scale.T
            # scale_u_output_nk = torch.ones((N_ROWS, 1), device=x_tensor_nc.device, dtype=torch.half) # Shape (N, 1)
            
            scale_u_output_nk = torch.mean(tensor_to_quantize_abs_nc, dim=1, keepdim=True) # Shape (N, 1)
            scale_u_output_nk = scale_u_output_nk / scale_u_output_nk.mean(dim=0, keepdim=True)
            scale_v_output_ck = mean_scale_c.unsqueeze(1).contiguous().to(torch.half) # Shape (C, 1)
        effective_rank = 1 # Kernel needs rank >= 1 for its loop
    else: # rank >= 1
        # Calculate rank-K approximation for scale based on abs(tensor_to_quantize)
        # subspace_iter(N, C) returns U(N, K), V_t(K, C)
        assert rank > 0, "Rank must be > 1 for rank-K approximation"
        with Profiler.scope(f"compact.quant.scale_rank{rank}_approx"):
            scale_U_nk, scale_V_t_kc, _ = subspace_iter(
                tensor_to_quantize_abs_nc, rank=rank, num_iters=2
            )
        # Kernel expects U(N, K) and V(C, K)
        scale_u_output_nk = scale_U_nk.contiguous().to(torch.half)       # Shape (N, K)
        scale_v_output_ck = scale_V_t_kc.transpose(0, 1).contiguous().to(torch.half) # Shape (C, K)
        effective_rank = rank
        assert scale_u_output_nk.shape == (N_ROWS, rank)
        assert scale_v_output_ck.shape == (C_COLS, rank)


    # Allocate outputs
    packed_output = torch.empty((N_ROWS, C_COLS_8), dtype=torch.uint8, device=x_tensor_nc.device)
    new_base_output_nc = torch.empty_like(x_tensor_nc) if update_cache else None

    BLOCK_SIZE_C = 512
    assert BLOCK_SIZE_C % 8 == 0, "BLOCK_SIZE_C must be divisible by 8"
    grid = (N_ROWS, triton.cdiv(C_COLS, BLOCK_SIZE_C))

    # Prepare dummy pointers/strides if not used
    dummy_tensor = x_tensor_nc # Use existing tensor for properties

    # New base pointers/strides (dummy if not update_cache)
    new_base_ptr = new_base_output_nc if update_cache else dummy_tensor
    stride_newb_n = new_base_ptr.stride(0) if update_cache else 0 # Stride for N dim
    stride_newb_c = new_base_ptr.stride(1) if update_cache else 0 # Stride for C dim


    with Profiler.scope("compact._binary_quant_fastpath"):
         _binary_quant_fastpath[grid](
             x_tensor_nc, base_tensor_nc,
             scale_u_output_nk, scale_v_output_ck, # Pass calculated U(N,K) and V(C,K)
             packed_output,
             new_base_ptr,
             # --- Dimensions (Passed as constexpr) ---
             N_ROWS=N_ROWS, C_COLS=C_COLS, C_COLS_8=C_COLS_8,
             RANK=effective_rank, # Pass the effective rank (1 for mean, K for subspace)
             # --- Strides (N, C Layout) ---
             stride_xn=x_tensor_nc.stride(0), stride_xc=x_tensor_nc.stride(1),
             stride_bn=base_tensor_nc.stride(0), stride_bc=base_tensor_nc.stride(1),
             stride_scale_un=scale_u_output_nk.stride(0), stride_scale_uk=scale_u_output_nk.stride(1),
             stride_scale_vc=scale_v_output_ck.stride(0), stride_scale_vk=scale_v_output_ck.stride(1),
             stride_packed_n=packed_output.stride(0), stride_packed_c8=packed_output.stride(1),
             stride_newb_n=stride_newb_n, stride_newb_c=stride_newb_c,
             # --- Meta-parameters (Passed as constexpr) ---
             BLOCK_SIZE_C=BLOCK_SIZE_C,
             UPDATE_CACHE=update_cache,
         )

    # Return values based on update_cache flag
    if update_cache:
        # Return 4 values: packed(N,C//8), u(N,K), v(C,K), new_base(N,C)
        return packed_output, scale_u_output_nk, scale_v_output_ck, new_base_output_nc
    else:
        # Always return 4 values, but last one is None if not update_cache
        return packed_output, scale_u_output_nk, scale_v_output_ck, None

# --- Simulation Functions ---

# Simulation uses slowpath quant/dequant OR manual packing/scaling for rank=-1
def sim_binary_quant_fastpath(
    x_tensor_nc: torch.Tensor,        # Input (N, C)
    base_tensor_nc: torch.Tensor,     # Input (N, C)
    rank: int,
    update_cache: bool,
):
    """
    Simulated version of binary_quant_fastpath.
    Uses slowpath quantize_1bit/dequantize_1bit internally, which handle N,C layout and rank=-1.
    Returns: packed(N, C//8), scale_u(N,K or N,1), scale_v(C,K or C,1), new_base(N,C)|None
    """
    assert rank >= 1 or rank == -1, "Rank must be >= 1 or -1"

    N, C = x_tensor_nc.shape
    new_base_nc = None

    # Calculate tensor to quantize (Always delta)
    tensor_to_quantize_nc = x_tensor_nc - base_tensor_nc

    # Use slowpath quantize_1bit - it now handles N,C layout and rank=-1
    # It returns packed(N, C//8), u(N, K or N, 1), v(K, C or 1, C)
    with Profiler.scope("compact.sim_quant.slowpath_quantize_1bit"):
        packed_sim, scale_u_nk, scale_v_kc = quantize_1bit(tensor_to_quantize_nc, rank=rank)

    # Determine the scales to return, matching the fastpath wrapper's expected V(C, K) layout
    # scale_v_kc from quantize_1bit is (K, C) or (1, C). We need (C, K) or (C, 1)
    scale_v_output_ck_sim = scale_v_kc.transpose(0, 1).contiguous() # Shape (C, K) or (C, 1)
    scale_u_output_nk_sim = scale_u_nk # Already (N, K) or (N, 1)

    if update_cache:
        # --- Perform Dequantization using slowpath dequantize_1bit --- 
        # It expects packed(N,C//8), u(N,K), v(K,C) and returns N,C
        with Profiler.scope("compact.sim_quant.slowpath_dequantize_1bit"):
            # Use the scales directly from the quantize_1bit call
            recv_quantized_nc = dequantize_1bit(packed_sim, scale_u_nk, scale_v_kc)

        # Calculate new base
        new_base_nc = base_tensor_nc + recv_quantized_nc

    # Return values matching fastpath signature: packed(N,C//8), u(N,K), v(C,K), new_base(N,C)
    return packed_sim, scale_u_output_nk_sim, scale_v_output_ck_sim, new_base_nc


@triton.jit
def _binary_dequant_fastpath(
    # Input Pointers
    packed_in_ptr,     # Packed delta (N, C//8) uint8
    scale_u_ptr,       # Scale factor u (N, K)
    scale_v_ptr,       # Scale factor v (C, K)
    base_ptr,          # Base cache (N, C) half
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (N, C) half
    # Dimensions
    N_ROWS: tl.constexpr,    # Original N
    C_COLS: tl.constexpr,    # Original C
    C_COLS_8: tl.constexpr,  # C_COLS // 8
    RANK: tl.constexpr,       # Effective rank K
    # Strides
    stride_packed_n, stride_packed_c8,
    stride_scale_un, stride_scale_uk, # Strides for U (N, K)
    stride_scale_vc, stride_scale_vk, # Strides for V (C, K)
    stride_base_n, stride_base_c,
    stride_recon_n, stride_recon_c,
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr, # Block size for C dimension
):
    """
    Dequantizes delta (packed along C dim) and calculates reconstructed activation (base + recv_delta).
    Scale factors U(N, K), V(C, K) are provided. Scale calculation dot(U[n,:], V[c,:]) happens INSIDE the kernel.
    Input: packed(N, C//8), u(N,K), v(C,K), base(N,C)
    Output: reconstructed(N, C)
    Grid: (N_ROWS, cdiv(C_COLS, BLOCK_SIZE_C))
    """
    pid_n = tl.program_id(0); pid_c_block = tl.program_id(1)
    c_block_start = pid_c_block * BLOCK_SIZE_C
    offs_c = c_block_start + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < C_COLS

    # --- Dequantize Block --- 
    # --- Load Scale Components U(N, K), V(C, K) --- 
    offs_k = tl.arange(0, RANK)
    
    # Load U vector for current row pid_n: shape (K,)
    scale_u_row_ptr = scale_u_ptr + pid_n * stride_scale_un
    scale_u_vec = tl.load(scale_u_row_ptr + offs_k * stride_scale_uk) # (RANK,)

    # Load V block for current C block: shape (BLOCK_SIZE_C, RANK)
    v_block_start_ptr = scale_v_ptr + c_block_start * stride_scale_vc
    # Offsets: (BLOCK_SIZE_C, 1) * stride_vc + (1, RANK) * stride_vk
    offs_v = tl.arange(0, BLOCK_SIZE_C)[:, None] * stride_scale_vc + offs_k[None, :] * stride_scale_vk
    # Mask for V load: (BLOCK_SIZE_C, 1) & (1, RANK)
    mask_v = mask_c[:, None] & (offs_k[None, :] < RANK)
    scale_v_block = tl.load(v_block_start_ptr + offs_v, mask=mask_v, other=0.0) # (BLOCK_SIZE_C, RANK)

    # --- Calculate Scale: dot(U[n,:], V[c,:]) -> sum(U[n,k] * V[c,k]) --- 
    scale_block = tl.sum(scale_u_vec[None, :] * scale_v_block, axis=1).to(tl.float16) # (BLOCK_SIZE_C,)

    # --- Load and unpack bits (packed along C) --- 
    c8_block_start = c_block_start // 8
    offs_c8 = c8_block_start + tl.arange(0, BLOCK_SIZE_C // 8)
    mask_c8 = offs_c8 < C_COLS_8
    
    # Load packed bytes for the current row
    packed_row_ptr = packed_in_ptr + pid_n * stride_packed_n
    
    # Map unpacked C offsets back to packed C//8 offsets
    byte_indices_in_row = offs_c // 8
    bit_indices_in_byte = offs_c % 8
    
    # Create mask for loading packed bytes (only need bytes relevant to current C block)
    # Use mask_c to determine valid elements, then find corresponding byte indices
    final_byte_mask = mask_c & (byte_indices_in_row < C_COLS_8) # Ensure byte index is valid
    
    # Load the necessary packed bytes. Need to handle potential non-contiguous loads efficiently.
    # A simple approach (might be less efficient): load all bytes corresponding to the C block elements
    packed_bytes_for_elems = tl.load(packed_row_ptr + byte_indices_in_row * stride_packed_c8, 
                                       mask=final_byte_mask, other=0)
                                       
    # Extract the correct bit for each element in the C block
    bits = ((packed_bytes_for_elems >> bit_indices_in_byte) & 1)

    # --- Calculate Reconstructed Delta --- 
    signs = tl.where(mask_c, (2 * bits - 1).to(tl.int8), 0) # Get (+1/-1) signs
    recv_quantized_block = signs * scale_block # This is recv_delta (BLOCK_SIZE_C,)

    # --- Load Base --- 
    base_row_ptr = base_ptr + pid_n * stride_base_n
    base_block = tl.load(base_row_ptr + offs_c * stride_base_c, mask=mask_c, other=0.0) # (BLOCK_SIZE_C,)

    # --- Calculate Output Block (Always level 1) --- 
    recon_block = base_block + recv_quantized_block # recon = base + recv_delta

    # --- Store Output Block --- 
    recon_out_row_ptr = recon_ptr + pid_n * stride_recon_n
    tl.store(recon_out_row_ptr + offs_c * stride_recon_c, recon_block, mask=mask_c)


@Profiler.prof_func("compact.binary_dequant_fastpath")
def binary_dequant_fastpath(
    packed: torch.Tensor,    # Input packed delta (N, C//8) uint8
    scale_u_nk: torch.Tensor,       # Input scale u (N, K)
    scale_v_ck: torch.Tensor,       # Input scale v (C, K)
    base_nc: torch.Tensor,         # Input base cache (N, C) half
    # rank: int, # Rank is inferred from scales now
):
    """
    Dequantizes delta (packed along C) and calculates reconstructed activation (base + recv_delta).
    Scale calculation dot(U[n,:], V[c,:]) happens INSIDE the kernel.
    Handles both rank-K and rank=-1 cases based on scale shapes.

    Input: packed(N, C//8), u(N,K), v(C,K), base(N,C)
    Output: reconstructed(N, C)
    """
    # Assertions
    assert packed.dtype == torch.uint8
    assert scale_u_nk.dtype == torch.half and scale_v_ck.dtype == torch.half
    assert base_nc.dtype == torch.half
    assert packed.ndim == 2 and scale_u_nk.ndim == 2 and scale_v_ck.ndim == 2 and base_nc.ndim == 2

    packed = packed.contiguous()
    scale_u_nk = scale_u_nk.contiguous()
    scale_v_ck = scale_v_ck.contiguous()
    base_nc = base_nc.contiguous()

    N_ROWS, C_COLS_8 = packed.shape
    C_COLS = C_COLS_8 * 8
    effective_rank = scale_u_nk.shape[1] # Infer rank from scale U
    assert effective_rank >= 1, "Inferred rank from scale_u must be >= 1"
    assert scale_v_ck.shape[1] == effective_rank, f"Scale V rank mismatch: V K dim {scale_v_ck.shape[1]} vs inferred K {effective_rank}"
    assert scale_v_ck.shape[0] == C_COLS, f"Scale V C dim mismatch: V C dim {scale_v_ck.shape[0]} vs expected C {C_COLS}"

    assert base_nc.shape == (N_ROWS, C_COLS), f"Base shape mismatch: {base_nc.shape} vs expected {(N_ROWS, C_COLS)}"
    assert scale_u_nk.shape == (N_ROWS, effective_rank), f"Scale U shape mismatch: {scale_u_nk.shape} vs expected {(N_ROWS, effective_rank)}"
    # V shape checked above

    # Allocate output tensors
    reconstructed_output_nc = torch.empty_like(base_nc)

    BLOCK_SIZE_C = 512
    assert BLOCK_SIZE_C % 8 == 0, "BLOCK_SIZE_C must be divisible by 8 for unpacking logic"
    grid = (N_ROWS, triton.cdiv(C_COLS, BLOCK_SIZE_C))

    # Prepare dummy pointers/strides if not used (not needed here)
    # dummy_tensor = base_nc # Use existing tensor for properties

    with Profiler.scope("compact._binary_dequant_fastpath"):
        _binary_dequant_fastpath[grid](
            packed,
            scale_u_nk, scale_v_ck, # Pass U(N,K), V(C,K)
            base_nc,
            reconstructed_output_nc,
            # --- Dimensions (Passed as constexpr) ---
            N_ROWS=N_ROWS, C_COLS=C_COLS, C_COLS_8=C_COLS_8,
            RANK=effective_rank, # Pass inferred effective rank
            # --- Strides (N, C Layout) ---
            stride_packed_n=packed.stride(0), stride_packed_c8=packed.stride(1),
            stride_scale_un=scale_u_nk.stride(0), stride_scale_uk=scale_u_nk.stride(1),
            stride_scale_vc=scale_v_ck.stride(0), stride_scale_vk=scale_v_ck.stride(1),
            stride_base_n=base_nc.stride(0), stride_base_c=base_nc.stride(1),
            stride_recon_n=reconstructed_output_nc.stride(0), stride_recon_c=reconstructed_output_nc.stride(1),
            # --- Meta-parameters (Passed as constexpr) ---
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

    # Return only recon
    return reconstructed_output_nc

# Simulation uses slowpath dequant OR manual dequant for rank=-1
def sim_binary_dequant_fastpath(
    packed_in_nc8: torch.Tensor, # Packed bits (N, C//8) UINT8
    scale_u_nk: torch.Tensor,     # Scale factor U (N, K or N, 1)
    scale_v_ck: torch.Tensor,     # Scale factor V (C, K or C, 1)
    base_nc: torch.Tensor,       # Base (N, C)
    # rank: int, # Rank is inferred
):
    """
    Simulated version of binary_dequant_fastpath.
    Uses the slowpath dequantize_1bit function internally.
    Handles both rank-K and rank-1 cases based on scale shapes.
    Input: packed(N, C//8), u(N,K), v(C,K), base(N,C)
    Returns: reconstructed(N,C)
    """
    # Infer rank from scales
    effective_rank = scale_u_nk.shape[1]
    assert effective_rank >= 1, "Inferred rank must be >= 1"
    assert scale_v_ck.shape[1] == effective_rank, "Scale V rank mismatch"

    N = scale_u_nk.shape[0]
    C = scale_v_ck.shape[0]
    assert packed_in_nc8.shape == (N, C // 8)
    assert scale_u_nk.shape == (N, effective_rank)
    assert scale_v_ck.shape == (C, effective_rank)
    assert base_nc.shape == (N, C)

    # --- Always use the slowpath dequantize_1bit for simulation consistency --- 
    # It handles rank=1 correctly when provided with appropriate rank-1 U/V scales.
    # slowpath dequantize_1bit expects u(N,K) and v(K,C).
    # We have u(N,K) and v(C,K) from fastpath.
    # Map: u_nk_slowpath = scale_u_nk, v_kc_slowpath = scale_v_ck.T
    with Profiler.scope("compact.sim_dequant.slowpath_dequantize_1bit"):
        u_nk_for_slowpath = scale_u_nk # Already (N, K)
        v_kc_for_slowpath = scale_v_ck.transpose(0, 1).contiguous() # Shape (K, C)

        recv_quantized_nc = dequantize_1bit(packed_in_nc8, u_nk_for_slowpath, v_kc_for_slowpath)
        # recv_quantized_nc is already (N, C)
        reconstructed_nc = base_nc + recv_quantized_nc

    # Return only reconstructed
    return reconstructed_nc

# --------- INT2 Fastpath Kernels ---------

@triton.jit
def _int2_quant_fastpath(
    # Input Pointers (Layout N, C)
    x_ptr,             # Current activation (N, C)
    base_ptr,          # Cached base (N, C)
    chan_scale_ptr,    # Channel scale (1, C) float16 - Calculated Externally
    tok_scale_ptr,     # Token scale (N, 1) float16 - Calculated Externally
    # Output Pointers
    packed_out_ptr,    # Packed indices (N, C//4) uint8 - OUTPUT
    new_base_ptr,      # Output new base (N, C) float16 - optional write
    # Dimensions
    N_ROWS: tl.constexpr,    # Original N
    C_COLS: tl.constexpr,    # Original C
    C_COLS_4: tl.constexpr,  # C_COLS // 4
    # Strides
    stride_xn, stride_xc,
    stride_bn, stride_bc,
    stride_chansc_c,   # stride for chan_scale C dim (stride_chansc_n is 0)
    stride_toksc_n,    # stride for tok_scale N dim (stride_toksc_c is 0)
    stride_packed_n, stride_packed_c4,
    stride_newb_n, stride_newb_c,      # Strides for new_base (N, C)
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr, # Block size for C dimension
    UPDATE_CACHE: tl.constexpr,
):
    """
    Quantizes delta (x - base) to 2-bit indices based on pre-calculated channel & token scales.
    Uses sign/magnitude bit logic matching quantize_int2.
    Packs four 2-bit indices into one uint8 along C dim. Output packed layout (N, C//4).
    Optionally updates base cache.
    Grid: (N_ROWS, cdiv(C_COLS, BLOCK_SIZE_C))
    """
    pid_n = tl.program_id(0); pid_c_block = tl.program_id(1)
    c_block_start = pid_c_block * BLOCK_SIZE_C
    offs_c = c_block_start + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < C_COLS

    # --- Load Inputs (Row pid_n, Block of C) ---
    x_row_ptr = x_ptr + pid_n * stride_xn
    base_row_ptr = base_ptr + pid_n * stride_bn
    x_block = tl.load(x_row_ptr + offs_c * stride_xc, mask=mask_c, other=0.0).to(tl.float16)
    base_block = tl.load(base_row_ptr + offs_c * stride_bc, mask=mask_c, other=0.0).to(tl.float16)

    # --- Calculate Delta ---
    delta_block = x_block - base_block # (BLOCK_SIZE_C,) float16

    # --- Calculate Scale Threshold (using pre-calculated scales) ---
    chan_scale_block_ptr = chan_scale_ptr + offs_c * stride_chansc_c
    chan_scale_block = tl.load(chan_scale_block_ptr, mask=mask_c, other=0.0).to(tl.float16)
    tok_scale_ptr_n = tok_scale_ptr + pid_n * stride_toksc_n
    tok_scale_scalar = tl.load(tok_scale_ptr_n).to(tl.float16)
    scale_threshold_block = (chan_scale_block * tok_scale_scalar).to(tl.float16)

    # --- Map Delta to Indices (0, 1, 2, 3) using sign/magnitude bit logic ---
    sign_bit = (delta_block >= 0).to(tl.uint8) # 0 for neg, 1 for non-neg
    magnitude_bit = (tl.abs(delta_block) > scale_threshold_block).to(tl.uint8) # 0 for small mag, 1 for large mag
    indices = (sign_bit << 1) | magnitude_bit # Combine bits: sign=bit1, mag=bit0
    # Indices map: 0(neg,small), 1(neg,large), 2(pos,small), 3(pos,large)
    indices = tl.where(mask_c, indices, 0) # Mask out invalid indices

    # --- Pack Indices (Pack along C) ---
    indices_reshaped = tl.reshape(indices, (BLOCK_SIZE_C // 4, 4))
    shifts = tl.arange(0, 4).to(tl.uint8) * 2 # [0, 2, 4, 6]
    shifted = (indices_reshaped << shifts).to(tl.uint8)
    packed_block = tl.sum(shifted, axis=1).to(tl.uint8)

    # --- Store Packed Block ---
    c4_block_start = c_block_start // 4
    offs_c4 = c4_block_start + tl.arange(0, BLOCK_SIZE_C // 4)
    mask_c4 = offs_c4 < C_COLS_4
    packed_output_row_ptr = packed_out_ptr + pid_n * stride_packed_n
    packed_output_ptrs = packed_output_row_ptr + offs_c4 * stride_packed_c4
    tl.store(packed_output_ptrs, packed_block, mask=mask_c4)

    # --- Update Cache (Conditional) ---
    if UPDATE_CACHE:
        # --- Map Indices back to Levels (avoiding multiple tl.where) ---
        # This logic directly maps the index bits back to the levels.
        # sign_bit = (indices >> 1)
        # magnitude_bit = indices & 1
        level_small_mag = 0.5 * scale_threshold_block
        level_large_mag = 2.0 * scale_threshold_block
        # Use magnitude bit to select base magnitude level
        magnitude_level = tl.where(magnitude_bit == 0, level_small_mag, level_large_mag)
        # Use sign bit to determine sign multiplier
        sign_multiplier = (sign_bit.to(tl.float16) * 2.0) - 1.0 # Maps 0-> -1, 1-> +1
        # Calculate reconstructed delta
        recv_delta = sign_multiplier * magnitude_level
        recv_delta = tl.where(mask_c, recv_delta, 0.0) # Mask out

        # Calculate new base
        new_base_block = base_block + recv_delta.to(tl.float16)

        # Store new base
        new_base_row_ptr = new_base_ptr + pid_n * stride_newb_n
        tl.store(new_base_row_ptr + offs_c * stride_newb_c, new_base_block, mask=mask_c)


@Profiler.prof_func("compact.int2_quant_fastpath")
def int2_quant_fastpath(
    x_tensor_nc: torch.Tensor,        # Input (N, C)
    base_tensor_nc: torch.Tensor,     # Input (N, C)
    update_cache: bool,
    rank: int = -1,                   # Add rank argument for interface consistency
):
    """
    Quantizes delta (x - base) to 2-bit using fast path kernel. Input/Output (N, C).
    Calculates channel and token scales based on delta.
    Matches binary interface by returning scales as U(N,1), V(C,1).

    Returns: packed(N, C//4), scale_u(N,1), scale_v(C,1), new_base(N,C)|None
    """
    # Assertions
    assert rank == -1, "INT2 fastpath only supports channel/token scales (rank=-1 equivalent)" # Enforce rank=-1
    assert x_tensor_nc.dtype == torch.half
    assert base_tensor_nc.dtype == torch.half
    assert x_tensor_nc.ndim == 2 and base_tensor_nc.ndim == 2
    assert x_tensor_nc.shape == base_tensor_nc.shape
    # Assert is_cuda removed for potential future CPU compatibility if needed
    # assert x_tensor_nc.is_cuda and base_tensor_nc.is_cuda

    x_tensor_nc = x_tensor_nc.contiguous()
    base_tensor_nc = base_tensor_nc.contiguous()

    N_ROWS, C_COLS = x_tensor_nc.shape
    assert C_COLS % 4 == 0, "C_COLS must be divisible by 4 for packing output alignment"
    C_COLS_4 = C_COLS // 4

    # --- Scale Calculation (Based on Delta, matches quantize_int2) ---
    with Profiler.scope("compact.quant.calc_scale_int2"):
        # Calculate delta directly in half precision
        delta_half = x_tensor_nc - base_tensor_nc 
        abs_delta = torch.abs(delta_half) # Now working with half precision
        chan_scale_1c = torch.mean(abs_delta, dim=0, keepdim=True) # Shape (1, C), half
        tok_scale_n1 = torch.mean(abs_delta, dim=1, keepdim=True)  # Shape (N, 1), half
        # Normalize tok_scale
        tok_mean = tok_scale_n1.mean()
        tok_scale_n1 = tok_scale_n1 / (tok_mean + 1e-6)
        # Scales are already half precision
        chan_scale_1c_half = chan_scale_1c.contiguous()
        tok_scale_n1_half = tok_scale_n1.contiguous()

    # Allocate outputs
    packed_output_nc4 = torch.empty((N_ROWS, C_COLS_4), dtype=torch.uint8, device=x_tensor_nc.device)
    new_base_output_nc = torch.empty_like(x_tensor_nc) if update_cache else None

    BLOCK_SIZE_C = 512 # Or tune this
    assert BLOCK_SIZE_C % 4 == 0, "BLOCK_SIZE_C must be divisible by 4"
    grid = (N_ROWS, triton.cdiv(C_COLS, BLOCK_SIZE_C))

    # New base pointers/strides (dummy if not update_cache)
    new_base_ptr = new_base_output_nc if update_cache else x_tensor_nc # Use existing tensor for properties if None
    stride_newb_n = new_base_ptr.stride(0) if update_cache else 0
    stride_newb_c = new_base_ptr.stride(1) if update_cache else 0

    # Kernel still needs (1,C) and (N,1) scales internally
    with Profiler.scope("compact._int2_quant_fastpath"):
         _int2_quant_fastpath[grid](
                     x_tensor_nc, base_tensor_nc,
             chan_scale_1c_half, tok_scale_n1_half, # Pass calculated scales (1,C), (N,1)
             packed_output_nc4,
             new_base_ptr,
             # --- Dimensions (Passed as constexpr) ---
             N_ROWS=N_ROWS, C_COLS=C_COLS, C_COLS_4=C_COLS_4,
             # --- Strides ---
             stride_xn=x_tensor_nc.stride(0), stride_xc=x_tensor_nc.stride(1),
             stride_bn=base_tensor_nc.stride(0), stride_bc=base_tensor_nc.stride(1),
             stride_chansc_c=chan_scale_1c_half.stride(1), # Stride N is 0 for chan_scale
             stride_toksc_n=tok_scale_n1_half.stride(0),    # Stride C is 0 for tok_scale
             stride_packed_n=packed_output_nc4.stride(0), stride_packed_c4=packed_output_nc4.stride(1),
             stride_newb_n=stride_newb_n, stride_newb_c=stride_newb_c,
             # --- Meta-parameters (Passed as constexpr) ---
             BLOCK_SIZE_C=BLOCK_SIZE_C,
             UPDATE_CACHE=update_cache,
         )

    # Prepare return values matching binary format (U(N,1), V(C,1))
    scale_u_nk = tok_scale_n1_half # Shape (N, 1)
    scale_v_ck = chan_scale_1c_half.view(C_COLS, 1) #  (1,C) -> (C,1)

    # Return values based on update_cache flag
    if update_cache:
        return packed_output_nc4, scale_u_nk, scale_v_ck, new_base_output_nc
    else:
        return packed_output_nc4, scale_u_nk, scale_v_ck, None

@triton.jit
def _int2_dequant_fastpath(
    # Input Pointers
    packed_in_ptr,     # Packed indices (N, C//4) uint8
    chan_scale_ptr,    # Channel scale (1, C) float16
    tok_scale_ptr,     # Token scale (N, 1) float16
    base_ptr,          # Base cache (N, C) float16
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (N, C) float16
    # Dimensions
    N_ROWS: tl.constexpr,    # Original N
    C_COLS: tl.constexpr,    # Original C
    C_COLS_4: tl.constexpr,  # C_COLS // 4
    # Strides
    stride_packed_n, stride_packed_c4,
    stride_chansc_c,   # stride for chan_scale C dim
    stride_toksc_n,    # stride for tok_scale N dim
    stride_base_n, stride_base_c,
    stride_recon_n, stride_recon_c,
    # Meta-parameters
    BLOCK_SIZE_C: tl.constexpr, # Block size for C dimension
):
    """
    Dequantizes 2-bit delta (packed along C dim) and calculates reconstructed activation (base + recv_delta).
    Scale calculation happens INSIDE the kernel using channel and token scales.
    Input: packed(N, C//4), chan_scale(1,C), tok_scale(N,1), base(N,C)
    Output: reconstructed(N, C)
    Grid: (N_ROWS, cdiv(C_COLS, BLOCK_SIZE_C))
    """
    pid_n = tl.program_id(0); pid_c_block = tl.program_id(1)
    c_block_start = pid_c_block * BLOCK_SIZE_C
    offs_c = c_block_start + tl.arange(0, BLOCK_SIZE_C)
    mask_c = offs_c < C_COLS

    # --- Load Base ---
    base_row_ptr = base_ptr + pid_n * stride_base_n
    base_block = tl.load(base_row_ptr + offs_c * stride_base_c, mask=mask_c, other=0.0).to(tl.float16)

    # --- Calculate Scale Threshold ---
    chan_scale_block_ptr = chan_scale_ptr + offs_c * stride_chansc_c
    chan_scale_block = tl.load(chan_scale_block_ptr, mask=mask_c, other=0.0).to(tl.float16)
    tok_scale_ptr_n = tok_scale_ptr + pid_n * stride_toksc_n
    tok_scale_scalar = tl.load(tok_scale_ptr_n).to(tl.float16)
    scale_threshold_block = (chan_scale_block * tok_scale_scalar).to(tl.float16)

    # --- Load and unpack bits (packed along C) ---
    byte_indices_in_row = offs_c // 4
    elem_indices_in_byte = offs_c % 4 # 0, 1, 2, 3
    final_byte_mask = mask_c & (byte_indices_in_row < C_COLS_4)
    packed_row_ptr = packed_in_ptr + pid_n * stride_packed_n
    packed_bytes_for_elems = tl.load(packed_row_ptr + byte_indices_in_row * stride_packed_c4,
                                       mask=final_byte_mask, other=0)
    shifts = (elem_indices_in_byte * 2).to(tl.uint8) # 0, 2, 4, 6
    indices = ((packed_bytes_for_elems >> shifts) & 0x03).to(tl.uint8)

    # --- Map Indices to Levels Efficiently ---
    sign_bit = (indices >> 1) # 0 for indices 0,1; 1 for indices 2,3
    magnitude_bit = indices & 1 # 0 for indices 0,2; 1 for indices 1,3
    level_small_mag = 0.5 * scale_threshold_block
    level_large_mag = 2.0 * scale_threshold_block
    magnitude_level = tl.where(magnitude_bit == 0, level_small_mag, level_large_mag)
    sign_multiplier = (sign_bit.to(tl.float16) * 2.0) - 1.0
    recv_delta = sign_multiplier * magnitude_level
    recv_delta = tl.where(mask_c, recv_delta, 0.0) # Mask out

    # --- Calculate Output Block ---
    recon_block = base_block + recv_delta.to(tl.float16)

    # --- Store Output Block ---
    recon_out_row_ptr = recon_ptr + pid_n * stride_recon_n
    tl.store(recon_out_row_ptr + offs_c * stride_recon_c, recon_block, mask=mask_c)


@Profiler.prof_func("compact.int2_dequant_fastpath")
def int2_dequant_fastpath(
    packed: torch.Tensor,    # Input packed indices (N, C//4) uint8
    scale_u_nk: torch.Tensor,       # Input scale u (N, 1) float16
    scale_v_ck: torch.Tensor,       # Input scale v (C, 1) float16---
    base_nc: torch.Tensor,          # Input base cache (N, C) float16
):
    """
    Dequantizes 2-bit delta (packed along C dim) and calculates reconstructed activation (base + recv_delta).
    Accepts scales in binary format U(N,K), V(C,K) but expects K=1 for INT2.
    Internally converts scales back to channel(1,C) and token(N,1) format for the kernel.

    Input: packed(N, C//4), scale_u(N,1), scale_v(C,1), base(N,C)
    Output: reconstructed(N, C)
    """
    # Assertions
    assert packed.dtype == torch.uint8
    assert scale_u_nk.dtype == torch.half and scale_v_ck.dtype == torch.half
    assert base_nc.dtype == torch.half
    assert packed.ndim == 2 and scale_u_nk.ndim == 2 and scale_v_ck.ndim == 2 and base_nc.ndim == 2

    # Check K=1 for INT2 scales
    assert scale_u_nk.shape[1] == 1, f"INT2 expects K=1 for scale_u, got shape {scale_u_nk.shape}"
    assert scale_v_ck.shape[1] == 1, f"INT2 expects K=1 for scale_v, got shape {scale_v_ck.shape}"
    # Assert is_cuda removed
    # assert packed_in_nc4.is_cuda and scale_u_nk.is_cuda and scale_v_ck.is_cuda and base_nc.is_cuda

    packed = packed.contiguous()
    scale_u_nk = scale_u_nk.contiguous()
    scale_v_ck = scale_v_ck.contiguous()
    base_nc = base_nc.contiguous()

    N_ROWS, C_COLS_4 = packed.shape
    C_COLS = C_COLS_4 * 4
    assert base_nc.shape == (N_ROWS, C_COLS), f"Base shape mismatch: {base_nc.shape} vs expected {(N_ROWS, C_COLS)}"
    assert scale_u_nk.shape == (N_ROWS, 1), f"Token scale (scale_u) shape mismatch: {scale_u_nk.shape} vs expected {(N_ROWS, 1)}"
    assert scale_v_ck.shape == (C_COLS, 1), f"Channel scale (scale_v) shape mismatch: {scale_v_ck.shape} vs expected {(C_COLS, 1)}"
    # --- Convert scales for the Kernel ---
    # Kernel expects chan_scale (1, C) and tok_scale (N, 1)
    tok_scale_n1_half = scale_u_nk # Already (N, 1)
    chan_scale_1c_half = scale_v_ck.transpose(0, 1) # Transpose (C, 1) -> (1, C)
    chan_scale_1c_half = chan_scale_1c_half.contiguous() # Ensure contiguous after transpose

    reconstructed_output_nc = torch.empty_like(base_nc)

    BLOCK_SIZE_C = 512 # Or tune this
    assert BLOCK_SIZE_C % 4 == 0, "BLOCK_SIZE_C must be divisible by 4 for unpacking logic"
    grid = (N_ROWS, triton.cdiv(C_COLS, BLOCK_SIZE_C))

    with Profiler.scope("compact._int2_dequant_fastpath"):
        _int2_dequant_fastpath[grid](
            packed,
            chan_scale_1c_half, tok_scale_n1_half, # Pass scales in kernel's expected format
            base_nc,
            reconstructed_output_nc,
            # --- Dimensions (Passed as constexpr) ---
            N_ROWS=N_ROWS, C_COLS=C_COLS, C_COLS_4=C_COLS_4,
            # --- Strides ---
            stride_packed_n=packed.stride(0), stride_packed_c4=packed.stride(1),
            stride_chansc_c=chan_scale_1c_half.stride(1), # Stride N is 0
            stride_toksc_n=tok_scale_n1_half.stride(0),   # Stride C is 0
            stride_base_n=base_nc.stride(0), stride_base_c=base_nc.stride(1),
            stride_recon_n=reconstructed_output_nc.stride(0), stride_recon_c=reconstructed_output_nc.stride(1),
            # --- Meta-parameters (Passed as constexpr) ---
            BLOCK_SIZE_C=BLOCK_SIZE_C,
        )

    return reconstructed_output_nc

# Simulation uses slowpath functions but needs matching interface
def sim_int2_quant_fastpath(
    x_tensor_nc: torch.Tensor,        # Input (N, C)
    base_tensor_nc: torch.Tensor,     # Input (N, C)
    update_cache: bool,
    rank: int = -1,                   # Add rank argument
):
    """
    Simulated version of int2_quant_fastpath. Matches binary interface.
    Uses slowpath quantize_int2/dequantize_int2 internally.
    Returns: packed(N, C//4), scale_u(N,1), scale_v(C,1), new_base(N,C)|None
    """
    assert rank == -1, "Simulation for INT2 fastpath only supports rank=-1 equivalent" # Enforce rank=-1
    N, C = x_tensor_nc.shape
    new_base_nc = None

    # Calculate tensor to quantize (Always delta)
    delta_nc = x_tensor_nc - base_tensor_nc

    # Use slowpath quantize_int2 - it handles N,C layout
    # It returns packed(N, C//4), chan_scale(1, C), tok_scale(N, 1)
    with Profiler.scope("compact.sim_quant.slowpath_quantize_int2"):
        packed_sim_nc4, chan_scale_sim_1c, tok_scale_sim_n1 = quantize_int2(delta_nc)

    if update_cache:
        # --- Perform Dequantization using slowpath dequantize_int2 --- 
        with Profiler.scope("compact.sim_quant.slowpath_dequantize_int2"):
            # Pass the scales calculated by quantize_int2
            recv_delta_sim_nc = dequantize_int2(packed_sim_nc4, chan_scale_sim_1c, tok_scale_sim_n1)
        
        # Calculate new base
        new_base_nc = base_tensor_nc + recv_delta_sim_nc

    # Prepare return values matching binary format (U(N,1), V(C,1))
    scale_u_nk_sim = tok_scale_sim_n1.to(torch.half) # Already (N, 1), ensure half
    scale_v_ck_sim = chan_scale_sim_1c.T.to(torch.half) # Transpose (1,C) -> (C,1), ensure half

    # Return values matching fastpath signature
    return packed_sim_nc4, scale_u_nk_sim, scale_v_ck_sim, new_base_nc

def sim_int2_dequant_fastpath(
    packed_in_nc4: torch.Tensor,    # Input packed indices (N, C//4) uint8
    scale_u_nk: torch.Tensor,       # Input scale u (N, 1) float16
    scale_v_ck: torch.Tensor,       # Input scale v (C, 1) float16
    base_nc: torch.Tensor,          # Input base cache (N, C) float16
):
    """
    Simulated version of int2_dequant_fastpath. Matches binary interface.
    Uses the slowpath dequantize_int2 function internally.
    Input: packed(N, C//4), scale_u(N,1), scale_v(C,1), base(N,C)
    Returns: reconstructed(N,C)
    """
    N, C4 = packed_in_nc4.shape
    C = C4 * 4
    assert scale_u_nk.shape == (N, 1), f"Sim expects scale_u shape {(N, 1)}, got {scale_u_nk.shape}"
    assert scale_v_ck.shape == (C, 1), f"Sim expects scale_v shape {(C, 1)}, got {scale_v_ck.shape}"
    assert base_nc.shape == (N, C)

    # --- Convert scales back for slowpath dequantize_int2 ---
    # slowpath expects chan_scale (1, C) and tok_scale (N, 1)
    tok_scale_n1_sim = scale_u_nk # Already (N, 1)
    chan_scale_1c_sim = scale_v_ck.T # Transpose (C, 1) -> (1, C)

    # --- Always use the slowpath dequantize_int2 for simulation consistency --- 
    with Profiler.scope("compact.sim_dequant.slowpath_dequantize_int2"):
        recv_delta_nc = dequantize_int2(packed_in_nc4, chan_scale_1c_sim, tok_scale_n1_sim)
        reconstructed_nc = base_nc + recv_delta_nc

    # Return only reconstructed
    return reconstructed_nc