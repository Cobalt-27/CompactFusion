import torch
import time
import functools

class Profiler:
    def __init__(self):
        self.events = {}
        self.enabled = True  # Track whether profiling is enabled

    def enable(self):
        """Enable profiling - all subsequent start/stop calls will be recorded."""
        self.enabled = True

    def disable(self):
        """Disable profiling - all subsequent start/stop calls will be ignored."""
        self.enabled = False

    def start(self, name, stream=None, cpu=False):
        """
        Start recording time for a named section. Supports multiple starts.
        No-op if profiling is disabled.
        """
        # Skip if profiling is disabled
        if not self.enabled:
            return

        if name not in self.events:
            self.events[name] = {'start': [], 'end': [], 'elapsed': 0.0, 'cpu': cpu}
        assert len(self.events[name]['start']) == len(self.events[name]['end']), \
            f"Cannot start '{name}' as there are more starts than stops"
        
        if cpu:
            start_event = time.time()
        else:
            start_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                start_event.record(stream)
            else:
                start_event.record()
        
        self.events[name]['start'].append(start_event)

    def stop(self, name, stream=None, cpu=False):
        """
        Stop recording time for a named section. Accumulates total time for multiple stops.
        No-op if profiling is disabled.
        """
        # Skip if profiling is disabled
        if not self.enabled:
            return

        assert name in self.events, f"No events recorded for '{name}'"
        assert len(self.events[name]['start']) - 1 == len(self.events[name]['end']), \
            f"Cannot stop '{name}' as there are more stops than starts"

        if cpu:
            end_event = time.time()
        else:
            end_event = torch.cuda.Event(enable_timing=True)
            if stream is not None:
                end_event.record(stream)
            else:
                end_event.record()
        
        self.events[name]['end'].append(end_event)

    def elapsed_time(self, name):
        """
        Get the total accumulated time for a specific named section.
        Syncs and stores the result, clearing start and end events after.
        """
        if name not in self.events:
            raise ValueError(f"No events recorded for '{name}'")
        
        # Check if CPU timing or CUDA event timing is used
        cpu = self.events[name]['cpu']
        total_time = self.events[name]['elapsed']
        total_count = len(self.events[name]['start'])
        # Accumulate new times
        if cpu:
            for start, end in zip(self.events[name]['start'], self.events[name]['end']):
                total_time += (end - start) * 1000  # Convert seconds to ms
        else:
            torch.cuda.synchronize()
            for start, end in zip(self.events[name]['start'], self.events[name]['end']):
                total_time += start.elapsed_time(end)

        # Store the accumulated time and clear the events
        self.events[name]['elapsed'] = total_time
        self.events[name]['start'] = []
        self.events[name]['end'] = []
        avg_time = total_time / total_count if total_count > 0 else 0
        return total_time, avg_time

    def get_all_elapsed_times(self):
        """
        Returns a dictionary of the accumulated elapsed times for each recorded event.
        """
        total_times = {}
        avg_times = {}
        for name in self.events:
            total_times[name], avg_times[name] = self.elapsed_time(name)
        return total_times, avg_times

    def sync(self):
        """
        Manually synchronize all recorded events.
        """
        torch.cuda.synchronize()
        self.get_all_elapsed_times() # flush all events

    def reset(self):
        """
        Reset all stored events.
        """
        self.sync()
        self.events = {}

    _instance = None

    @staticmethod
    def instance():
        """
        Singleton pattern to get the instance of the profiler.
        """
        if Profiler._instance is None:
            Profiler._instance = Profiler()
        return Profiler._instance
    
    class ProfileContext:
        def __init__(self, profiler, name, stream=None, cpu=False):
            self.profiler = profiler
            self.name = name
            self.stream = stream
            self.cpu = cpu

        def __enter__(self):
            self.profiler.start(self.name, self.stream, cpu=self.cpu)

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.profiler.stop(self.name, self.stream, cpu=self.cpu)
    
    @staticmethod
    def scope(name, stream=None, cpu=False):
        """
        Create a context manager for profiling a block of code.
        Usage:
        with CudaProfiler.scope('name', cpu=True):
            # Code to profile
        """
        prof = Profiler.instance()
        return prof.ProfileContext(prof, name, stream, cpu)

    @staticmethod
    def prof_func(name, cpu=False):
        """
        Decorator to profile a function using the CudaProfiler.
        Logs the total execution time of the function.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Start profiling
                Profiler.instance().start(name, cpu=cpu)
                result = func(*args, **kwargs)
                # Stop profiling
                Profiler.instance().stop(name, cpu=cpu)
                return result
            return wrapper
        return decorator

def prof_summary(profiler: Profiler, rank = None):
    """
    prof result breakdown
    """
    if rank is None:
        rank='N/A'
    total_times, avg_times = profiler.get_all_elapsed_times()
    total_time, _ = profiler.elapsed_time('total')
    split = "-" * 20
    output_lines = []
    output_lines.append(split)
    output_lines.append(f"⚒️    Profiling Summary for Rank {rank}")
    # sort by time
    for key, total_times in sorted(total_times.items(), key=lambda item: item[1], reverse=True):
        line = f"🔷 [Rank {rank}] [{key}] {total_times/1000:.2f}s {total_times/total_time:.2%} avg={avg_times[key]:.2f}ms"
        output_lines.append(line)
    output_lines.append(split)
    return output_lines


_torch_profiler = None


def torch_profiler_step():
    global _torch_profiler
    if _torch_profiler is not None:
        _torch_profiler.step()


def set_torch_profiler(profiler):
    global _torch_profiler
    _torch_profiler = profiler