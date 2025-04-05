import torch
import pytest
from xfuser.prof import Profiler

@pytest.mark.parametrize("seed", [42, 43])
def test_profiler_with_manual_and_profiler_events(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    # Create CUDA events for manual timing
    manual_start_A = torch.cuda.Event(enable_timing=True)
    manual_end_A = torch.cuda.Event(enable_timing=True)
    manual_start_ABC = torch.cuda.Event(enable_timing=True)
    manual_end_ABC = torch.cuda.Event(enable_timing=True)
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    # Start manual timing for the full sequence ABC
    manual_start_ABC.record()
    # Perform operation A and time it manually
    manual_start_A.record()
    profiler.start('op_AB')
    
    torch.matmul(x, y)  # Operation A
    manual_end_A.record()

    # Start profiler for operation AB
    

    # Perform operation B
    torch.matmul(x, y)  # Operation B

    # Stop profiler for operation AB
    profiler.stop('op_AB')

    # Perform operation C
    torch.matmul(x, y)  # Operation C

    # End manual timing for the full sequence ABC
    manual_end_ABC.record()

    # Sync CUDA events
    torch.cuda.synchronize()

    # Get manually measured times
    manual_time_A = manual_start_A.elapsed_time(manual_end_A)
    manual_time_ABC = manual_start_ABC.elapsed_time(manual_end_ABC)

    # Get profiler measured time for AB
    profiler_time_AB, _ = profiler.elapsed_time('op_AB')

    # Validate the inequalities: A < AB < ABC
    assert manual_time_A < profiler_time_AB, \
        f"Failed: Manual time for A ({manual_time_A} ms) should be less than profiler time for AB ({profiler_time_AB} ms)"
    assert profiler_time_AB < manual_time_ABC, \
        f"Failed: Profiler time for AB ({profiler_time_AB} ms) should be less than manual time for ABC ({manual_time_ABC} ms)"

    print("Test 1 (A < AB < ABC) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_multiple_sections(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    # Record timings for two separate operations (X and Y)
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # Start profiling operation X
    profiler.start('op_X')
    torch.matmul(x, y)  # Operation X
    profiler.stop('op_X')

    # Start profiling operation Y
    profiler.start('op_Y')
    torch.matmul(x, y)  # Operation Y
    profiler.stop('op_Y')

    # Get elapsed times for both operations
    profiler_time_X, _ = profiler.elapsed_time('op_X')
    profiler_time_Y, _ = profiler.elapsed_time('op_Y')

    # Ensure both timings are greater than zero
    assert profiler_time_X > 0, "Failed: Profiler time for X should be greater than 0"
    assert profiler_time_Y > 0, "Failed: Profiler time for Y should be greater than 0"

    print("Test 2 (Multiple Sections) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_overlapping_sections(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    # Record timings for overlapping operations (P and PQ)
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # Start profiling operation P
    profiler.start('op_PQ')
    torch.matmul(x, y)  # Operation P

    # Start profiling operation PQ (includes P and Q)
    profiler.start('op_P')
    torch.matmul(x, y)  # Operation Q
    profiler.stop('op_P')

    # Stop profiling operation P (after both P and Q are done)
    profiler.stop('op_PQ')

    # Get elapsed times for both operations
    profiler_time_P, _ = profiler.elapsed_time('op_P')
    profiler_time_PQ, _ = profiler.elapsed_time('op_PQ')

    # Ensure P < PQ, as PQ includes both P and Q
    assert profiler_time_P < profiler_time_PQ, \
        f"Failed: Profiler time for P ({profiler_time_P} ms) should be less than profiler time for PQ ({profiler_time_PQ} ms)"

    print("Test 3 (Overlapping Sections) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_manual_sync(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    # Record timings for an operation
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # Start profiling and record without stopping
    profiler.start('op_sync')
    torch.matmul(x, y)  # Operation
    profiler.stop('op_sync')

    # Manually sync CUDA events
    profiler.sync()

    # Get elapsed time after syncing
    profiler_time_sync, _ = profiler.elapsed_time('op_sync')

    # Ensure elapsed time is greater than zero
    assert profiler_time_sync > 0, \
        f"Failed: Profiler time after manual sync ({profiler_time_sync} ms) should be greater than 0"

    print("Test 4 (Manual Sync) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_profiler_accuracy(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    # Create CUDA events for manual timing
    manual_start = torch.cuda.Event(enable_timing=True)
    manual_end = torch.cuda.Event(enable_timing=True)
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # Start manual timing
    manual_start.record()
    profiler.start('op_accuracy')

    # Perform operation
    for _ in range(100):
        temp = torch.matmul(x, y)

    # Stop manual timing
    manual_end.record()
    profiler.stop('op_accuracy')

    # Sync CUDA events
    torch.cuda.synchronize()

    # Get manually measured time
    manual_time = manual_start.elapsed_time(manual_end)

    # Get profiler measured time
    profiler_time, _ = profiler.elapsed_time('op_accuracy')


    # Ensure the profiler time is within 5% of the manual time
    error_margin = 0.05 * manual_time
    assert abs(profiler_time - manual_time) <= error_margin, \
        f"Failed: Profiler time ({profiler_time} ms) should be within 5% of manual time ({manual_time} ms)"

    print("Test 5 (Profiler Accuracy) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_profiler_enable_disable(seed):
    torch.manual_seed(seed)
    profiler = Profiler()

    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # First operation - profiling is enabled (default)
    profiler.start('op_enabled')
    torch.matmul(x, y)
    profiler.stop('op_enabled')

    # Disable profiling
    profiler.disable()

    # Second operation - profiling is disabled, should not be recorded
    profiler.start('op_disabled')
    torch.matmul(x, y)
    profiler.stop('op_disabled')

    # Third operation with same name as first - should also not be recorded
    profiler.start('op_enabled')
    torch.matmul(x, y)
    profiler.stop('op_enabled')

    # Re-enable profiling
    profiler.enable()

    # Fourth operation - profiling is enabled again
    profiler.start('op_reenabled')
    torch.matmul(x, y)
    profiler.stop('op_reenabled')

    # Get elapsed times for the operations
    profiler_time_enabled, _ = profiler.elapsed_time('op_enabled')
    
    # Check if 'op_disabled' was recorded
    op_disabled_recorded = 'op_disabled' in profiler.events
    
    profiler_time_reenabled, _ = profiler.elapsed_time('op_reenabled')

    # Ensure only the enabled operations were recorded
    assert profiler_time_enabled > 0, "Failed: Enabled operation should have been recorded"
    assert not op_disabled_recorded, "Failed: Disabled operation should not have been recorded"
    assert profiler_time_reenabled > 0, "Failed: Re-enabled operation should have been recorded"

    print("Test 6 (Profiler Enable/Disable) passed!")

@pytest.mark.parametrize("seed", [42, 43])
def test_profiler_context_with_enable_disable(seed):
    torch.manual_seed(seed)
    # Use the singleton instance instead of creating a new one
    profiler = Profiler.instance()
    
    # Reset the profiler to start with a clean state
    profiler.reset()
    
    # Make sure profiling is enabled
    profiler.enable()

    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')

    # Use context manager with profiling enabled
    with Profiler.scope('op_context_enabled'):
        torch.matmul(x, y)

    # Disable profiling
    profiler.disable()

    # Use context manager with profiling disabled
    with Profiler.scope('op_context_disabled'):
        torch.matmul(x, y)

    # Re-enable profiling
    profiler.enable()

    # Use context manager with profiling re-enabled
    with Profiler.scope('op_context_reenabled'):
        torch.matmul(x, y)

    # Check if operations were recorded correctly
    op_enabled_recorded = 'op_context_enabled' in profiler.events
    op_disabled_recorded = 'op_context_disabled' in profiler.events
    op_reenabled_recorded = 'op_context_reenabled' in profiler.events

    # Ensure only the enabled operations were recorded
    assert op_enabled_recorded, "Failed: Operation with context manager (enabled) should have been recorded"
    assert not op_disabled_recorded, "Failed: Operation with context manager (disabled) should not have been recorded"
    assert op_reenabled_recorded, "Failed: Operation with context manager (re-enabled) should have been recorded"

    print("Test 7 (Profiler Context Manager with Enable/Disable) passed!")
