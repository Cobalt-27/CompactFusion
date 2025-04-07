import torch
import numpy as np
import random
import os # Added for path operations

class StatsLogger:
    """Simple statistics logger for compression metrics."""
    
    def __init__(self):
        # Main storage for stats
        self.stats = {}
        # Prev step storage for similarity calculations
        self.prev_activations = {}
        self.prev_deltas = {}
        self.plot_counter = 0 # for plotting
        # For volume tracking
        self.total_original_volume = 0
        self.total_compressed_volume = 0
        # Track steps per key for filenames
        self.step_counts = {} 

    def log(
        self,
        key, 
        base, 
        delta_base, 
        before_comp_activation, 
        recv_activation, 
        compressed_tensor, 
        compress_residual,
        ref_activation_path: str | None = None, 
        dump_activations: bool = False,
        calc_total_error: bool = False
    ):
        """
        Log compression statistics for a layer.
        
        Args:
            key: String identifier for the layer
            base: Base activation used for delta calculation
            delta_base: Delta base used for delta-delta calculation (only for residual=2)
            before_comp_activation: Activation before compression step
            recv_activation: Reconstructed activation after compression
            compressed_tensor: The tensor after compression, need decoding for real activation
            compress_residual: Residual compression level (0, 1, or 2)
            ref_activation_path (str | None): Path for dumping/loading reference activations.
            dump_activations (bool): If True and path is set, dump activations.
            calc_total_error (bool): If True and path is set, calculate error against reference.
        """
        
        # Increment step count for this key
        step_count = self.step_counts.get(key, 0)
        self.step_counts[key] = step_count + 1
        
        # --- Dump Activations (assert path if flag is True) ---
        if dump_activations:
            # Assert that the path is provided if dumping is requested
            assert ref_activation_path is not None, \
                "ref_activation_path must be provided when dump_activations is True"
            dump_dir = ref_activation_path
            os.makedirs(dump_dir, exist_ok=True)
            filename = os.path.join(dump_dir, f"{key}_step{step_count}.pt")
            torch.save(before_comp_activation.detach().cpu(), filename)

        # Initialize if first time
        if key not in self.stats:
            self.stats[key] = []
        
        # Calculate on-the-fly compression error
        error = torch.norm(before_comp_activation - recv_activation)
        
        # --- Compare with Dumped Activations (assert path if flag is True) ---
        total_error = None
        if calc_total_error:
            # Assert that the path is provided if calculation is requested
            assert ref_activation_path is not None, \
                "ref_activation_path must be provided when calc_total_error is True"
            load_dir = ref_activation_path
            filename = os.path.join(load_dir, f"{key}_step{step_count}.pt")
            # Let it crash if file not found
            gt_activation = torch.load(filename, map_location='cpu')
            total_error = torch.norm(recv_activation.cpu() - gt_activation)

        # Calculate sizes
        original_size_bytes = before_comp_activation.numel() * before_comp_activation.element_size()
        compressed_size_bytes = compressed_tensor.numel() * compressed_tensor.element_size()
        
        # Accumulate total volumes
        self.total_original_volume += original_size_bytes
        self.total_compressed_volume += compressed_size_bytes
        
        # Calculate delta and delta-delta based on residual level
        if compress_residual == 0:
            delta = None
            delta_delta = None
        elif compress_residual == 1:
            delta = before_comp_activation - base 
            delta_delta = None
        elif compress_residual == 2:
            delta = before_comp_activation - base
            delta_delta = before_comp_activation - base - delta_base
            # from xfuser.compact.plot import plot_3d
            # plot_3d(delta_delta, title=f"dd_{key}_{self.plot_counter}")
            # self.plot_counter += 1
        else:
            raise ValueError('invalid residual')
            
        # Calculate norms
        act_norm = torch.norm(before_comp_activation).item()
        delta_norm = torch.norm(delta).item() if delta is not None else None
        delta_delta_norm = torch.norm(delta_delta).item() if delta_delta is not None else None
        
        # Calculate similarities with previous step
        act_sim = None
        delta_sim = None
        
        if key in self.prev_activations:
            # Ensure tensors are flat and on the same device for similarity
            act_sim = torch.nn.functional.cosine_similarity(
                before_comp_activation.flatten(), 
                self.prev_activations[key].flatten().to(before_comp_activation.device), 
                dim=0
            ).item()
            
        if delta is not None and key in self.prev_deltas:
             delta_sim = torch.nn.functional.cosine_similarity(
                delta.flatten(), 
                self.prev_deltas[key].flatten().to(delta.device), 
                dim=0
            ).item()
        
        # Store current stats
        self.stats[key].append({
            'error': error.item(),
            'total_error': total_error.item() if total_error is not None else None, # Store total error
            'activation_norm': act_norm,
            'delta_norm': delta_norm,
            'delta_delta_norm': delta_delta_norm,
            'activation_similarity': act_sim,
            'delta_similarity': delta_sim,
            'residual': compress_residual,
            'original_size_bytes': original_size_bytes,
            'compressed_size_bytes': compressed_size_bytes,
        })
        
        # Store current activations and deltas for next step similarity (on CPU to save GPU memory)
        self.prev_activations[key] = before_comp_activation.detach().cpu()
        if delta is not None:
            self.prev_deltas[key] = delta.detach().cpu()
    
    def summary_over_steps(self, steps=None, keys=None):
        """
        Print a summary of the logged statistics over steps.
        
        Args:
            steps: List of step indices to include (None for all)
            keys: List of layer keys to summarize (None for all)
        """
        if not self.stats:
            print("No statistics logged yet.")
            return
        
        # Determine available keys
        available_keys = keys if keys is not None else self.stats.keys()
        
        # Find max number of steps across all keys
        max_steps = max([len(self.stats[k]) for k in available_keys if k in self.stats], default=0)
        
        # Determine which steps to report
        if steps is None:
            steps = list(range(max_steps))
        
        # For each step, use summary_over_keys with a single-step range
        for step in steps:
            if step >= max_steps:
                print(f"Step {step} is out of range")
                continue
                
            print(f"=== Step {step} ===")
            # Handle each key individually or pass None to show all keys
            if keys is None:
                self.summary_over_keys(step_range=(step, step+1), key=None)
            else:
                # If keys is a list, iterate through each key
                if isinstance(keys, list):
                    for k in keys:
                        self.summary_over_keys(step_range=(step, step+1), key=k)
                else:
                    # If keys is a single key, pass it directly
                    self.summary_over_keys(step_range=(step, step+1), key=keys)
    
    def summary_over_keys(self, step_range=None, key=None):
        """
        Print a summary of the logged statistics.
        
        Args:
            step_range: Range of steps to include (None for all)
            key: Layer key to summarize (None for all)
        """
        if not self.stats:
            print("No statistics logged yet.")
            return
            
        keys = [key] if key else self.stats.keys()
        
        for k in keys:
            if k not in self.stats:
                print(f"No data for key {k}")
                continue
                
            stats_list = self.stats[k]
            if step_range:
                stats_list = stats_list[step_range[0]:step_range[1]]
                
            if not stats_list:
                print(f"No data for key {k} in step range {step_range}")
                continue
                
            # Group by residual level
            by_residual = {}
            for stat in stats_list:
                res = stat['residual']
                if res not in by_residual:
                    by_residual[res] = []
                by_residual[res].append(stat)
            
            for res, stats in by_residual.items():
                print(f"🔵 [{k}] res={res} (over {len(stats)} steps):")
                
                # Compute averages
                avg_error = np.mean([s['error'] for s in stats])
                avg_total_error_list = [s['total_error'] for s in stats if s['total_error'] is not None]
                avg_total_error = np.mean(avg_total_error_list) if avg_total_error_list else None
                avg_act_norm = np.mean([s['activation_norm'] for s in stats])
                
                # Print first line with error and activation norm
                print(f"err: {avg_error:.3f}" + (f", total_err: {avg_total_error:.3f}" if avg_total_error is not None else ""), end="")
                print(f", act={avg_act_norm:.3f}", end="")

                if res >= 1:
                    avg_delta_norm = np.mean([s['delta_norm'] for s in stats if s['delta_norm'] is not None])
                    delta_ratio = avg_delta_norm/avg_act_norm
                    print(f", delta={avg_delta_norm:.3f}, d/a={delta_ratio:.2f}", end="")
                
                if res >= 2:
                    avg_delta_delta_norm = np.mean([s['delta_delta_norm'] for s in stats if s['delta_delta_norm'] is not None])
                    if avg_delta_norm > 0:
                        dd_ratio = avg_delta_delta_norm/avg_delta_norm
                        print(f", dd={avg_delta_delta_norm:.3f}, dd/d={dd_ratio:.2f}", end="")
                
                print()
                
                # Print second line with similarities on same line if available
                similarities = []
                
                act_sims = [s['activation_similarity'] for s in stats if s['activation_similarity'] is not None]
                if act_sims:
                    avg_act_sim = np.mean(act_sims)
                    similarities.append(f"act_sim: {avg_act_sim:.3f}")
                
                if res >= 1:
                    delta_sims = [s['delta_similarity'] for s in stats if s['delta_similarity'] is not None]
                    if delta_sims:
                        avg_delta_sim = np.mean(delta_sims)
                        similarities.append(f"delta_sim: {avg_delta_sim:.3f}")
                
                if similarities:
                    print(", ".join(similarities))
    
    def summary_compression_volume(self):
        """Prints the total data volume before and after compression and the ratio."""
        if self.total_original_volume == 0:
            print("💾 No volume data logged yet.") # Keep emoji
            return

        orig_mb = self.total_original_volume / (1024**2)
        comp_mb = self.total_compressed_volume / (1024**2)
        
        summary_line = f"💾 Vol: Orig {orig_mb:.2f} MB"
        summary_line += f", Comp {comp_mb:.2f} MB"
        
        if self.total_compressed_volume > 0:
            ratio = self.total_original_volume / self.total_compressed_volume
            summary_line += f", Ratio {ratio:.2f}x"
        else:
            summary_line += ", Ratio N/A"
            
        print(summary_line)

    def summary_total_avg(self):
        
        # Calculate average activation norm across all layers
        mean_act = np.mean([np.mean([s['activation_norm'] for s in stats]) for stats in self.stats.values()])
        
        # Calculate average delta norm (for residual >= 1)
        delta_values = []
        for stats_list in self.stats.values():
            for s in stats_list:
                if s['residual'] >= 1 and s['delta_norm'] is not None:
                    delta_values.append(s['delta_norm'])
        mean_delta = np.mean(delta_values) if delta_values else None
        
        # Calculate average delta-delta norm (for residual >= 2)
        dd_values = []
        for stats_list in self.stats.values():
            for s in stats_list:
                if s['residual'] >= 2 and s['delta_delta_norm'] is not None:
                    dd_values.append(s['delta_delta_norm'])
        mean_dd = np.mean(dd_values) if dd_values else None
        
        # Print all averages on one line
        print(f"🟫 avg activation: {mean_act:.3f}" + 
              (f", avg delta: {mean_delta:.3f}" if mean_delta is not None else "") + 
              (f", avg delta-delta: {mean_dd:.3f}" if mean_dd is not None else ""))
        
        mean_err = np.mean([np.mean([s['error'] for s in stats]) for stats in self.stats.values()])
        
        # Calculate average total error if available
        total_err_values = []
        for stats_list in self.stats.values():
            for s in stats_list:
                if s['total_error'] is not None:
                    total_err_values.append(s['total_error'])
        mean_total_err = np.mean(total_err_values) if total_err_values else None

        print(f"🟧 avg comp error: {mean_err:.3f}" + (f", avg total err: {mean_total_err:.3f}" if mean_total_err is not None else ", [total err not logged]"))
        from xfuser.compact.utils import get_emoji
        print(get_emoji())

# Global stats instance
_stats = None

def stats_log():
    global _stats
    if _stats is None:
        _stats = StatsLogger()
    return _stats

def stats_clear():
    global _stats
    _stats = None

def stats_verbose(step_range=None, key=None, summary_keys=True):
    if _stats is None:
        print("No statistics logged.")
        return
    if summary_keys:
        _stats.summary_over_keys(step_range, key)
    _stats.summary_compression_volume()
    _stats.summary_total_avg()

def log(key, base, delta_base, real_activation, recv_activation, compressed_tensor, compress_residual):
    """
    Global function to log compression statistics.
    
    Args:
        key: String identifier for the layer
        base: Base activation used for delta calculation
        delta_base: Delta base used for delta-delta calculation
        real_activation: Original activation without compression
        recv_activation: Reconstructed activation after compression
        compressed_tensor: The tensor after compression
        compress_residual: Residual compression level (0, 1, or 2)
    """
    stats_log().log(key, base, delta_base, real_activation, recv_activation, compressed_tensor, compress_residual)

def stats_verbose_steps(steps=None, keys=None):
    """
    Print a verbose summary of statistics for specific steps.
    
    Args:
        steps: List of step indices to include (None for all)
        keys: List of layer keys to summarize (None for all)
    """
    if _stats is None:
        print("No statistics logged.")
        return
    _stats.summary_over_steps(steps, keys)
