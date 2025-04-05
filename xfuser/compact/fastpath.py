import triton
import triton.language as tl
import torch
from xfuser.prof import Profiler
from xfuser.compact.compress_quantize import dequantize_1bit
from xfuser.compact.compress_quantize import quantize_1bit

@triton.jit
def _bitwise_or(a, b):
    """ Helper for Triton reduce """
    return a | b

@triton.jit
def _binary_quant_fastpath(
    # Input Pointers (Layout C, N)
    x_ptr,             # Current activation (C, N)
    base_ptr,          # Cached base (C, N)
    delta_base_ptr,    # Cached delta_base (C, N)
    # Output Pointers
    packed_dd_ptr,     # Packed delta_delta (C, N//8)
    scale_dd_ptr,      # Scale of delta_delta (C,)
    new_base_ptr,      # Output new base (C, N) - optional write
    new_delta_base_ptr,# Output new delta_base (C, N) - optional write
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    # Strides (ALL are for C, N layout, except packed)
    stride_xc, stride_xn,
    stride_bc, stride_bn,
    stride_dbc, stride_dbn,
    stride_packed_c, stride_packed_n8, # Strides for packed (C, N//8)
    stride_newb_c, stride_newb_n,      # Strides for new_base (C, N)
    stride_newdb_c, stride_newdb_n,    # Strides for new_delta_base (C, N)
    # Meta-parameters
    UPDATE_CACHE: tl.constexpr,
    DELTA_DECAY: tl.constexpr, # Decay factor for delta_base
):
    """
    Optimized kernel quantizing delta_delta using vectorized packing.
    Calculates: delta_delta = x - base - delta_base
    Packs into 1-bit representation using sign.
    
    Inputs: x, base, delta_base (all C, N)
    Outputs: packed_dd (C, N//8), scale_dd (C,)
    Optional: Computes cache updates if UPDATE_CACHE=True
    Grid: (CHANNEL,)
    """
    c_idx = tl.program_id(0)
    n_offsets = tl.arange(0, N_TOKENS)
    n8_offsets = tl.arange(0, N_TOKENS_8)

    # Load Full Rows [N_TOKENS]
    x_row_base_ptr = x_ptr + c_idx * stride_xc
    base_row_base_ptr = base_ptr + c_idx * stride_bc
    db_row_base_ptr = delta_base_ptr + c_idx * stride_dbc
    x_row = tl.load(x_row_base_ptr + n_offsets * stride_xn)
    base_row = tl.load(base_row_base_ptr + n_offsets * stride_bn)
    db_row = tl.load(db_row_base_ptr + n_offsets * stride_dbn)

    # Calculate delta_delta and scale
    delta_delta_row = x_row - base_row - db_row # [N_TOKENS]
    scale_val = tl.sum(tl.abs(delta_delta_row.to(tl.float32)), axis=0) / N_TOKENS
    scale_val_fp16 = scale_val.to(tl.float16)
    tl.store(scale_dd_ptr + c_idx, scale_val_fp16)

    # Calculate Signs
    binary = (delta_delta_row >= 0).to(tl.uint8) # [N_TOKENS]

    # Vectorized Packing [N_TOKENS] -> [N_TOKENS_8]
    binary_reshaped = tl.reshape(binary, (N_TOKENS_8, 8)) # [N_TOKENS_8, 8]
    shifts = tl.arange(0, 8).to(tl.uint8) # [8]
    shifted = (binary_reshaped << shifts).to(tl.uint8) # [N_TOKENS_8, 8]
    packed_dd = tl.reduce(shifted, axis=1, combine_fn=_bitwise_or).to(tl.uint8) # [N_TOKENS_8]

    # Store Packed Row
    packed_output_indices = c_idx * stride_packed_c + n8_offsets * stride_packed_n8
    tl.store(packed_dd_ptr + packed_output_indices, packed_dd)

    # Update Cache (when enabled)
    if UPDATE_CACHE:
        sign_int8 = (2 * binary.to(tl.int8) - 1) # [N_TOKENS]
        recv_delta_delta_row = sign_int8 * scale_val_fp16 # [N_TOKENS]
        recon_x_row = base_row + db_row + recv_delta_delta_row 

        new_base_indices = c_idx * stride_newb_c + n_offsets * stride_newb_n
        tl.store(new_base_ptr + new_base_indices, recon_x_row)

        new_delta_base_row = (db_row + recv_delta_delta_row) * DELTA_DECAY
        new_delta_base_indices = c_idx * stride_newdb_c + n_offsets * stride_newdb_n
        tl.store(new_delta_base_ptr + new_delta_base_indices, new_delta_base_row)


@Profiler.prof_func("compact.binary_quant_fastpath")
def binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    delta_base_tensor_cn: torch.Tensor, # Input (C, N)
    update_cache: bool,
    delta_decay_factor: float,  # Decay factor for delta_base
):
    """
    Quantize delta_delta to 1-bit representation with fast path kernel.
    
    Input: x, base, delta_base (C, N)
    Output: packed_dd (C, N//8), scale_dd (C,), 
            new_base and new_delta_base if update_cache=True, else None, None
    """
    assert x_tensor_cn.dtype == torch.half
    assert base_tensor_cn.dtype == torch.half
    assert delta_base_tensor_cn.dtype == torch.half
    assert x_tensor_cn.ndim == 2 and base_tensor_cn.ndim == 2 and delta_base_tensor_cn.ndim == 2
    assert x_tensor_cn.shape == base_tensor_cn.shape == delta_base_tensor_cn.shape
    assert x_tensor_cn.is_cuda and base_tensor_cn.is_cuda and delta_base_tensor_cn.is_cuda
    
    x_tensor_cn = x_tensor_cn.contiguous()
    base_tensor_cn = base_tensor_cn.contiguous()
    delta_base_tensor_cn = delta_base_tensor_cn.contiguous()

    CHANNEL, N_TOKENS = x_tensor_cn.shape
    assert N_TOKENS % 8 == 0, "N_TOKENS must be divisible by 8"
    N_TOKENS_8 = N_TOKENS // 8

    # Allocate outputs
    packed_dd_output = torch.empty((CHANNEL, N_TOKENS_8), dtype=torch.uint8, device=x_tensor_cn.device)
    scale_dd_output = torch.empty((CHANNEL,), dtype=torch.half, device=x_tensor_cn.device)
    new_base_output_cn = torch.empty_like(x_tensor_cn) if update_cache else None
    new_delta_base_output_cn = torch.empty_like(x_tensor_cn) if update_cache else None

    grid = (CHANNEL,)

    with Profiler.scope("compact._binary_quant_fastpath"):
         _binary_quant_fastpath[grid](
             x_tensor_cn, base_tensor_cn, delta_base_tensor_cn,
             packed_dd_output, scale_dd_output,
             new_base_output_cn if update_cache else x_tensor_cn,
             new_delta_base_output_cn if update_cache else x_tensor_cn,
            N_TOKENS, CHANNEL, N_TOKENS_8,
             x_tensor_cn.stride(0), x_tensor_cn.stride(1),
             base_tensor_cn.stride(0), base_tensor_cn.stride(1),
             delta_base_tensor_cn.stride(0), delta_base_tensor_cn.stride(1),
             packed_dd_output.stride(0), packed_dd_output.stride(1),
             new_base_output_cn.stride(0) if update_cache else 0,
             new_base_output_cn.stride(1) if update_cache else 0,
             new_delta_base_output_cn.stride(0) if update_cache else 0,
             new_delta_base_output_cn.stride(1) if update_cache else 0,
            UPDATE_CACHE=update_cache,
             DELTA_DECAY=float(delta_decay_factor),
        )

    if update_cache:
        return packed_dd_output, scale_dd_output, new_base_output_cn, new_delta_base_output_cn
    else:
        return packed_dd_output, scale_dd_output, None, None

def sim_binary_quant_fastpath(
    x_tensor_cn: torch.Tensor,        # Input (C, N)
    base_tensor_cn: torch.Tensor,     # Input (C, N)
    delta_base_tensor_cn: torch.Tensor, # Input (C, N)
    update_cache: bool,
    delta_decay_factor: float,
):
    """
    Simulated version of binary_quant_fastpath using PyTorch helpers.
    Handles transpositions internally to use N,C helpers.
    """
    # Calculate delta_delta (still C, N)
    delta_delta_cn = x_tensor_cn - base_tensor_cn - delta_base_tensor_cn

    # Transpose to (N, C) for quantize_1bit helper
    delta_delta_nc = delta_delta_cn.transpose(0, 1).contiguous()
    packed_dd, scale_dd = quantize_1bit(delta_delta_nc)
    # quantize_1bit outputs: packed (C, N//8), scale (C,) - Matches kernel output shapes

    if update_cache:
        # Dequantize requires packed (C, N//8) and scale (C,)
        # dequantize_1bit outputs N, C
        recv_delta_delta_nc = dequantize_1bit(packed_dd, scale_dd)

        # Transpose base and delta_base to N, C for calculation
        base_nc = base_tensor_cn.transpose(0, 1).contiguous()
        delta_base_nc = delta_base_tensor_cn.transpose(0, 1).contiguous()

        # Calculate new base and delta_base in N, C layout
        new_base_nc = base_nc + delta_base_nc + recv_delta_delta_nc
        new_delta_base_nc = (delta_base_nc + recv_delta_delta_nc) * delta_decay_factor

        # Transpose results back to C, N
        new_base_cn = new_base_nc.transpose(0, 1).contiguous()
        new_delta_base_cn = new_delta_base_nc.transpose(0, 1).contiguous()
        return packed_dd, scale_dd, new_base_cn, new_delta_base_cn
    else:
        return packed_dd, scale_dd, None, None


# --- Fused Dequantization + Reconstruction + New Delta Base Calculation (C, N Layout) ---

@triton.jit
def _binary_dequant_fastpath(
    # Input Pointers
    packed_dd_ptr,     # Packed delta_delta (C, N//8) uint8
    scale_dd_ptr,      # Scale of delta_delta (C,) half
    base_ptr,          # Base cache (C, N) half
    delta_base_ptr,    # Delta base cache (C, N) half
    # Output Pointers
    recon_ptr,         # Output reconstructed activation (C, N) half
    new_db_ptr,        # Output new delta_base (C, N) half
    # Dimensions
    N_TOKENS: tl.constexpr, # Original N
    CHANNEL: tl.constexpr,  # Original C
    N_TOKENS_8: tl.constexpr, # N_TOKENS // 8
    # Strides (ALL are for C, N layout, except packed)
    stride_packed_c, stride_packed_n8,
    stride_base_c, stride_base_n,
    stride_db_c, stride_db_n,
    stride_recon_c, stride_recon_n,
    stride_newdb_c, stride_newdb_n,
    # Meta-parameters
    DELTA_DECAY: tl.constexpr, # Decay factor for delta_base
):
    """
    Optimized dequantization kernel with loop-free unpacking.
    Calculates:
      - recv_delta_delta = dequantize(packed_dd, scale_dd)
      - reconstructed = base + delta_base + recv_delta_delta
      - new_delta_base = (delta_base + recv_delta_delta) * DELTA_DECAY
    
    Inputs: packed_dd (C, N//8), scale_dd (C,), base/delta_base (C, N) 
    Outputs: reconstructed (C, N), new_delta_base (C, N)
    Grid: (CHANNEL,)
    """
    c_idx = tl.program_id(0)
    n_offsets = tl.arange(0, N_TOKENS)

    # Load Scale [1]
    scale = tl.load(scale_dd_ptr + c_idx).to(tl.float16)

    # Dequantize to Full recv_delta_delta_row [N_TOKENS] 
    byte_indices = n_offsets // 8
    bit_indices = n_offsets % 8
    packed_row_base_ptr = packed_dd_ptr + c_idx * stride_packed_c
    packed_byte_ptrs = packed_row_base_ptr + byte_indices * stride_packed_n8
    packed_bytes_for_elems = tl.load(packed_byte_ptrs) # [N_TOKENS]
    bits = (packed_bytes_for_elems >> bit_indices) & 1 # [N_TOKENS]
    signs = (2 * bits - 1).to(tl.int8) # [N_TOKENS]
    recv_delta_delta_row = signs * scale # [N_TOKENS]

    # Load Base and Delta Base [N_TOKENS]
    base_row_ptr = base_ptr + c_idx * stride_base_c + n_offsets * stride_base_n
    db_row_ptr = delta_base_ptr + c_idx * stride_db_c + n_offsets * stride_db_n
    base_row = tl.load(base_row_ptr)
    db_row = tl.load(db_row_ptr)

    # Calculate Outputs [N_TOKENS]
    recon_row = base_row + db_row + recv_delta_delta_row
    new_db_row = (db_row + recv_delta_delta_row) * DELTA_DECAY

    # Store Outputs [N_TOKENS]
    recon_out_ptr = recon_ptr + c_idx * stride_recon_c + n_offsets * stride_recon_n
    new_db_out_ptr = new_db_ptr + c_idx * stride_newdb_c + n_offsets * stride_newdb_n
    tl.store(recon_out_ptr, recon_row)
    tl.store(new_db_out_ptr, new_db_row)


@Profiler.prof_func("compact.binary_dequant_fastpath")
def binary_dequant_fastpath(
    packed_dd_cn8: torch.Tensor,     # Input (C, N//8) uint8
    scale_dd_c: torch.Tensor,      # Input (C,) half
    base_cn: torch.Tensor,         # Input (C, N) half
    delta_base_cn: torch.Tensor, # Input (C, N) half
    delta_decay_factor: float,  # Decay factor for delta_base
):
    """
    Dequantize 1-bit values and calculate reconstructed output and new_delta_base using Triton kernel.
    
    Input: packed_dd (C, N//8), scale_dd (C,), base/delta_base (C, N)
    Output: reconstructed (C, N), new_delta_base (C, N)
    """
    assert packed_dd_cn8.dtype == torch.uint8
    assert scale_dd_c.dtype == torch.half
    assert base_cn.dtype == torch.half
    assert delta_base_cn.dtype == torch.half
    assert packed_dd_cn8.ndim == 2 and scale_dd_c.ndim == 1 and base_cn.ndim == 2 and delta_base_cn.ndim == 2
    assert packed_dd_cn8.shape[0] == scale_dd_c.shape[0] == base_cn.shape[0] == delta_base_cn.shape[0]
    assert packed_dd_cn8.shape[1] * 8 == base_cn.shape[1] == delta_base_cn.shape[1]
    assert packed_dd_cn8.is_cuda and scale_dd_c.is_cuda and base_cn.is_cuda and delta_base_cn.is_cuda

    packed_dd_cn8 = packed_dd_cn8.contiguous()
    scale_dd_c = scale_dd_c.contiguous()
    base_cn = base_cn.contiguous()
    delta_base_cn = delta_base_cn.contiguous()

    CHANNEL, N_TOKENS_8 = packed_dd_cn8.shape
    N_TOKENS = N_TOKENS_8 * 8
    assert base_cn.shape == (CHANNEL, N_TOKENS)

    # Allocate output tensors
    reconstructed_output_cn = torch.empty_like(base_cn)
    new_delta_base_output_cn = torch.empty_like(delta_base_cn)

    grid = (CHANNEL,)

    with Profiler.scope("compact._binary_dequant_fastpath"):
         _binary_dequant_fastpath[grid](
             packed_dd_cn8, scale_dd_c, base_cn, delta_base_cn,
             reconstructed_output_cn, new_delta_base_output_cn,
             N_TOKENS, CHANNEL, N_TOKENS_8,
             packed_dd_cn8.stride(0), packed_dd_cn8.stride(1),
             base_cn.stride(0), base_cn.stride(1),
             delta_base_cn.stride(0), delta_base_cn.stride(1),
             reconstructed_output_cn.stride(0), reconstructed_output_cn.stride(1),
             new_delta_base_output_cn.stride(0), new_delta_base_output_cn.stride(1),
             DELTA_DECAY=float(delta_decay_factor),
         )

    return reconstructed_output_cn, new_delta_base_output_cn

def sim_binary_dequant_fastpath(
    packed_dd_cn8: torch.Tensor,     # Input (C, N//8) uint8
    scale_dd_c: torch.Tensor,      # Input (C,) half
    base_cn: torch.Tensor,         # Input (C, N) half
    delta_base_cn: torch.Tensor, # Input (C, N) half
    delta_decay_factor: float,
):
    """
    Simulated version of binary_dequant_fastpath using PyTorch helpers.
    Handles transpositions internally to use N,C helpers.
    """
    # Dequantize needs packed (C, N//8), scale (C,) -> outputs (N, C)
    recv_delta_delta_nc = dequantize_1bit(packed_dd_cn8, scale_dd_c)

    # Transpose base and delta_base to N, C for calculation
    base_nc = base_cn.transpose(0, 1).contiguous()
    delta_base_nc = delta_base_cn.transpose(0, 1).contiguous()

    # Calculate reconstructed and new_delta_base in N, C layout
    reconstructed_nc = base_nc + delta_base_nc + recv_delta_delta_nc
    new_delta_base_nc = (delta_base_nc + recv_delta_delta_nc) * delta_decay_factor

    # Transpose results back to C, N
    reconstructed_cn = reconstructed_nc.transpose(0, 1).contiguous()
    new_delta_base_cn = new_delta_base_nc.transpose(0, 1).contiguous()

    return reconstructed_cn, new_delta_base_cn


# Update Test Function
def profile_quantize_kernels(num_runs=1000, num_warmup=5):
    """Profile kernel assuming C,N inputs/outputs."""
    import time
    from typing import Tuple, Optional

    N_TOKENS, CHANNEL = 4096, 4096
    DELTA_DECAY_FACTOR = 0.5  # Example decay factor for testing
    
    # Create original (N, C) tensors for reference calculation
    x_tensor_nc = torch.randn((N_TOKENS, CHANNEL), dtype=torch.half, device="cuda").contiguous()
    base_tensor_nc = (torch.randn_like(x_tensor_nc) * 0.1).contiguous()
    delta_base_tensor_nc = (torch.randn_like(x_tensor_nc) * 0.05).contiguous()

    # --- Calculate Reference Values (using PyTorch helpers on N,C layout) ---
    delta_delta_ref_nc = x_tensor_nc - base_tensor_nc - delta_base_tensor_nc
    ref_packed_dd, ref_scale_dd = quantize_1bit(delta_delta_ref_nc) # -> C, N//8 and C
    ref_recv_delta_delta_nc = dequantize_1bit(ref_packed_dd, ref_scale_dd) # -> N, C
    ref_new_base_nc = base_tensor_nc + delta_base_tensor_nc + ref_recv_delta_delta_nc # -> N, C
    ref_new_delta_base_nc = (delta_base_tensor_nc + ref_recv_delta_delta_nc) * DELTA_DECAY_FACTOR # -> N, C

    # Transpose references to (C, N) for comparison with fused kernel/sim output
    ref_new_base_cn = ref_new_base_nc.transpose(0, 1).contiguous()
    ref_new_delta_base_cn = ref_new_delta_base_nc.transpose(0, 1).contiguous()
    # Packed and scale refs are already in correct shape (C, N//8), (C,)
    # --- Reference Calculation End ---

    # --- Prepare (C, N) inputs for the fused kernel & sim function ---
    x_tensor_cn = x_tensor_nc.transpose(0, 1).contiguous()
    base_tensor_cn = base_tensor_nc.transpose(0, 1).contiguous()
    delta_base_tensor_cn = delta_base_tensor_nc.transpose(0, 1).contiguous()

    # --- Warm-up runs (Kernel & Sim) ---
    print(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        _ = binary_quant_fastpath(x_tensor_cn, base_tensor_cn, delta_base_tensor_cn, update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR)
        _ = sim_binary_quant_fastpath(x_tensor_cn, base_tensor_cn, delta_base_tensor_cn, update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR)
        torch.cuda.synchronize()

    print("Warmup complete. Starting performance measurement...")

    # --- Get results for correctness check (run once with update_cache=True) ---
    outputs_kernel_check = binary_quant_fastpath(
        x_tensor_cn, base_tensor_cn, delta_base_tensor_cn, 
        update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR
    )
    outputs_sim_check = sim_binary_quant_fastpath(
        x_tensor_cn, base_tensor_cn, delta_base_tensor_cn, 
        update_cache=True, delta_decay_factor=DELTA_DECAY_FACTOR
    )
    torch.cuda.synchronize()

    # --- Profiling Kernel ---
    print(f"  Profiling binary_quant_fastpath (Kernel, C,N with decay)...")
    torch.cuda.synchronize()
    start_kernel = time.time()
    for i in range(num_runs):
        update_c = (i % 2 == 0)
        # Run kernel but discard output
        _ = binary_quant_fastpath(
            x_tensor_cn + (i*0.001), 
            base_tensor_cn + (i*0.001), 
            delta_base_tensor_cn + (i*0.001), 
            update_cache=update_c,
            delta_decay_factor=DELTA_DECAY_FACTOR
        )
    torch.cuda.synchronize()
    end_kernel = time.time()
    fused_time = (end_kernel - start_kernel) / num_runs

    # --- Profiling Simulation ---
    print(f"  Profiling sim_binary_quant_fastpath (Simulation, C,N with decay)...")
    torch.cuda.synchronize()
    start_sim = time.time()
    for i in range(num_runs):
        update_c = (i % 2 == 0)
        # Run simulation function but discard output
        _ = sim_binary_quant_fastpath(
            x_tensor_cn + (i*0.001), 
            base_tensor_cn + (i*0.001), 
            delta_base_tensor_cn + (i*0.001), 
            update_cache=update_c,
            delta_decay_factor=DELTA_DECAY_FACTOR
        )
    torch.cuda.synchronize()
    end_sim = time.time()
    sim_time = (end_sim - start_sim) / num_runs


    # --- Verify correctness (using the single check run results) ---
    torch.cuda.synchronize()
    atol, rtol = 1e-3, 1e-2 # Keep slightly relaxed tolerance for FP16

    # Verify Kernel
    fused_packed_dd, fused_scale_dd, fused_new_base_cn, fused_new_delta_base_cn = outputs_kernel_check
    if fused_packed_dd is None: # Should not happen as we ran with update_cache=True
        print("Error: Kernel check run did not produce outputs.")
        kernel_correct = False
        k_scale_ok, k_packed_ok, k_new_base_ok, k_new_db_ok = False, False, False, False
    else:
        k_scale_ok = torch.allclose(ref_scale_dd, fused_scale_dd, atol=atol, rtol=rtol)
        k_packed_ok = torch.equal(ref_packed_dd, fused_packed_dd) # Packed should be exact
        k_new_base_ok = torch.allclose(ref_new_base_cn, fused_new_base_cn, atol=atol, rtol=rtol)
        k_new_db_ok = torch.allclose(ref_new_delta_base_cn, fused_new_delta_base_cn, atol=atol, rtol=rtol)
        kernel_correct = k_scale_ok and k_packed_ok and k_new_base_ok and k_new_db_ok

    # Verify Simulation
    sim_packed_dd, sim_scale_dd, sim_new_base_cn, sim_new_delta_base_cn = outputs_sim_check
    if sim_packed_dd is None: # Should not happen as we ran with update_cache=True
        print("Error: Simulation check run did not produce outputs.")
        sim_correct = False
        s_scale_ok, s_packed_ok, s_new_base_ok, s_new_db_ok = False, False, False, False
    else:
        s_scale_ok = torch.allclose(ref_scale_dd, sim_scale_dd, atol=atol, rtol=rtol)
        s_packed_ok = torch.equal(ref_packed_dd, sim_packed_dd) # Packed should be exact
        s_new_base_ok = torch.allclose(ref_new_base_cn, sim_new_base_cn, atol=atol, rtol=rtol)
        s_new_db_ok = torch.allclose(ref_new_delta_base_cn, sim_new_delta_base_cn, atol=atol, rtol=rtol)
        sim_correct = s_scale_ok and s_packed_ok and s_new_base_ok and s_new_db_ok

    # --- Print results ---
    print("--- Quant Performance --- (ms per run)")
    print(f"Kernel (C,N with decay): {fused_time*1000:.3f}")
    print(f"Sim (C,N with decay):    {sim_time*1000:.3f}")
    print("---")
    print(f"Correctness vs Ref (Kernel): {kernel_correct}")
    if not kernel_correct:
        print(f"  Scale OK: {k_scale_ok}, Packed OK: {k_packed_ok}, NewBase OK: {k_new_base_ok}, NewDeltaBase OK: {k_new_db_ok}")
        if not k_scale_ok: print(f"    Max kernel scale diff: {torch.max(torch.abs(ref_scale_dd - fused_scale_dd))}")
        if not k_packed_ok: print(f"    Kernel packed tensors differ.")
        if not k_new_base_ok: print(f"    Max kernel new_base_cn diff: {torch.max(torch.abs(ref_new_base_cn - fused_new_base_cn))}")
        if not k_new_db_ok: print(f"    Max kernel new_delta_base_cn diff: {torch.max(torch.abs(ref_new_delta_base_cn - fused_new_delta_base_cn))}")
    print(f"Correctness vs Ref (Sim):    {sim_correct}")
    if not sim_correct:
        print(f"  Scale OK: {s_scale_ok}, Packed OK: {s_packed_ok}, NewBase OK: {s_new_base_ok}, NewDeltaBase OK: {s_new_db_ok}")
        if not s_scale_ok: print(f"    Max sim scale diff: {torch.max(torch.abs(ref_scale_dd - sim_scale_dd))}")
        if not s_packed_ok: print(f"    Sim packed tensors differ.")
        if not s_new_base_ok: print(f"    Max sim new_base_cn diff: {torch.max(torch.abs(ref_new_base_cn - sim_new_base_cn))}")
        if not s_new_db_ok: print(f"    Max sim new_delta_base_cn diff: {torch.max(torch.abs(ref_new_delta_base_cn - sim_new_delta_base_cn))}")
    print("---")

    return fused_time, sim_time

# --- Test Function for Dequantization Kernel ---
def profile_dequantize_kernels(num_runs=1000, num_warmup=5):
    """Profile fused dequantization kernel assuming C,N inputs/outputs."""
    import time
    from typing import Tuple

    N_TOKENS, CHANNEL = 4096, 4096
    DELTA_DECAY_FACTOR = 0.5 # Use the same decay factor as quantize test for consistency

    # --- Generate Realistic Inputs ---
    # Create original (N, C) tensors first
    delta_delta_ref_nc = torch.randn((N_TOKENS, CHANNEL), dtype=torch.half, device="cuda") * 0.01 # Small values for DD
    base_ref_nc = (torch.randn((N_TOKENS, CHANNEL), dtype=torch.half, device="cuda") * 0.1).contiguous()
    delta_base_ref_nc = (torch.randn((N_TOKENS, CHANNEL), dtype=torch.half, device="cuda") * 0.05).contiguous()

    # Quantize DD using reference to get packed/scale inputs for dequant tests
    ref_packed_dd, ref_scale_dd = quantize_1bit(delta_delta_ref_nc) # -> C, N//8 and C

    # --- Calculate Reference Outputs (using PyTorch helpers on N,C layout) ---
    ref_recv_delta_delta_nc = dequantize_1bit(ref_packed_dd, ref_scale_dd) # -> N, C
    ref_reconstructed_nc = base_ref_nc + delta_base_ref_nc + ref_recv_delta_delta_nc # -> N, C
    ref_new_delta_base_nc = (delta_base_ref_nc + ref_recv_delta_delta_nc) * DELTA_DECAY_FACTOR # -> N, C

    # Transpose references to (C, N) for comparison with kernel/sim output
    ref_reconstructed_cn = ref_reconstructed_nc.transpose(0, 1).contiguous()
    ref_new_delta_base_cn = ref_new_delta_base_nc.transpose(0, 1).contiguous()
    # --- Reference Calculation End ---

    # --- Prepare (C, N) inputs for the fused dequant kernel & sim ---
    # Packed and scale are already (C, N//8), (C,)
    base_cn = base_ref_nc.transpose(0, 1).contiguous()
    delta_base_cn = delta_base_ref_nc.transpose(0, 1).contiguous()

    # --- Warm-up runs (Kernel & Sim) ---
    print(f"Running {num_warmup} warmup iterations for dequant kernel & sim...")
    for _ in range(num_warmup):
        _ = binary_dequant_fastpath(
            ref_packed_dd, ref_scale_dd, base_cn, delta_base_cn, delta_decay_factor=DELTA_DECAY_FACTOR
        )
        _ = sim_binary_dequant_fastpath(
            ref_packed_dd, ref_scale_dd, base_cn, delta_base_cn, delta_decay_factor=DELTA_DECAY_FACTOR
        )
        torch.cuda.synchronize()

    print("Warmup complete. Starting dequant performance measurement...")

    # --- Get results for correctness check (run once) ---
    outputs_kernel_check = binary_dequant_fastpath(
        ref_packed_dd, ref_scale_dd, base_cn, delta_base_cn, delta_decay_factor=DELTA_DECAY_FACTOR
    )
    outputs_sim_check = sim_binary_dequant_fastpath(
        ref_packed_dd, ref_scale_dd, base_cn, delta_base_cn, delta_decay_factor=DELTA_DECAY_FACTOR
    )
    torch.cuda.synchronize()

    # --- Profiling Kernel ---
    print(f"  Profiling binary_dequant_fastpath (Kernel, C,N)...")
    torch.cuda.synchronize()
    start_kernel = time.time()
    for i in range(num_runs):
        # Run kernel but discard output
        _ = binary_dequant_fastpath(
            ref_packed_dd,
            ref_scale_dd,
            base_cn, # Could add noise here if needed: base_cn + (i*0.0001),
            delta_base_cn, # delta_base_cn + (i*0.0001)
            delta_decay_factor=DELTA_DECAY_FACTOR
        )
    torch.cuda.synchronize()
    end_kernel = time.time()
    kernel_time = (end_kernel - start_kernel) / num_runs

    # --- Profiling Simulation ---
    print(f"  Profiling sim_binary_dequant_fastpath (Simulation, C,N)...")
    torch.cuda.synchronize()
    start_sim = time.time()
    for i in range(num_runs):
        # Run simulation but discard output
        _ = sim_binary_dequant_fastpath(
            ref_packed_dd,
            ref_scale_dd,
            base_cn,
            delta_base_cn,
            delta_decay_factor=DELTA_DECAY_FACTOR
        )
    torch.cuda.synchronize()
    end_sim = time.time()
    sim_time = (end_sim - start_sim) / num_runs

    # --- Verify correctness (using the single check run results) ---
    torch.cuda.synchronize()
    atol, rtol = 1e-3, 1e-2 # Keep slightly relaxed tolerance for FP16

    # Verify Kernel
    fused_reconstructed_cn, fused_new_delta_base_cn = outputs_kernel_check
    k_recon_ok = torch.allclose(ref_reconstructed_cn, fused_reconstructed_cn, atol=atol, rtol=rtol)
    k_new_db_ok = torch.allclose(ref_new_delta_base_cn, fused_new_delta_base_cn, atol=atol, rtol=rtol)
    kernel_correct = k_recon_ok and k_new_db_ok

    # Verify Simulation
    sim_reconstructed_cn, sim_new_delta_base_cn = outputs_sim_check
    s_recon_ok = torch.allclose(ref_reconstructed_cn, sim_reconstructed_cn, atol=atol, rtol=rtol)
    s_new_db_ok = torch.allclose(ref_new_delta_base_cn, sim_new_delta_base_cn, atol=atol, rtol=rtol)
    sim_correct = s_recon_ok and s_new_db_ok

    # --- Print results ---
    print("--- Dequant Performance --- (ms per run)")
    print(f"Kernel Dequant (C,N): {kernel_time*1000:.3f}")
    print(f"Sim Dequant (C,N):    {sim_time*1000:.3f}")
    print("---")
    print(f"Correctness vs Ref (Kernel): {kernel_correct}")
    if not kernel_correct:
        print(f"  Recon OK: {k_recon_ok}, New Delta Base OK: {k_new_db_ok}")
        if not k_recon_ok: print(f"    Max kernel reconstructed_cn diff: {torch.max(torch.abs(ref_reconstructed_cn - fused_reconstructed_cn))}")
        if not k_new_db_ok: print(f"    Max kernel new_delta_base_cn diff: {torch.max(torch.abs(ref_new_delta_base_cn - fused_new_delta_base_cn))}")
    print(f"Correctness vs Ref (Sim):    {sim_correct}")
    if not sim_correct:
        print(f"  Recon OK: {s_recon_ok}, New Delta Base OK: {s_new_db_ok}")
        if not s_recon_ok: print(f"    Max sim reconstructed_cn diff: {torch.max(torch.abs(ref_reconstructed_cn - sim_reconstructed_cn))}")
        if not s_new_db_ok: print(f"    Max sim new_delta_base_cn diff: {torch.max(torch.abs(ref_new_delta_base_cn - sim_new_delta_base_cn))}")
    print("---")

    return kernel_time, sim_time


if __name__ == "__main__":
    profile_quantize_kernels()
    profile_dequantize_kernels()
