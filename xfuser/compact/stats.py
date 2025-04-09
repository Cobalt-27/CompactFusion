import torch
import numpy as np
import random
import os # Added for path operations
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

EIGENVALUES_PLOT_STEPS = [1]
EIGENVALUES_PLOT_LAYERS = [0, 1]

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
        # Store eigenvalues for analysis
        self.eigenvalues = {}

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
        
        layer_idx = int(key.split('-')[0])
        if self.step_counts[key] in EIGENVALUES_PLOT_STEPS and layer_idx in EIGENVALUES_PLOT_LAYERS:
            if key not in self.eigenvalues:
                self.eigenvalues[key] = {'activation': [], 'delta': [], 'delta_delta': []}
        
            act_eigenvalues = self._compute_eigenvalues(before_comp_activation)
            self.eigenvalues[key]['activation'].append(act_eigenvalues)
            
            if delta is not None:
                delta_eigenvalues = self._compute_eigenvalues(delta)
                self.eigenvalues[key]['delta'].append(delta_eigenvalues)
            
            if delta_delta is not None:
                delta_delta_eigenvalues = self._compute_eigenvalues(delta_delta)
                self.eigenvalues[key]['delta_delta'].append(delta_delta_eigenvalues)
        
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
    
    def _compute_eigenvalues(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Compute eigenvalues of a tensor using SVD.
        
        Args:
            tensor: Input tensor to analyze
            
        Returns:
            Array of singular values (equivalent to eigenvalues for symmetric matrices)
        """
        tensor_cpu = tensor.detach().cpu()
        
        original_shape = tensor_cpu.shape
        if len(original_shape) > 2:
            tensor_2d = tensor_cpu.reshape(-1, original_shape[-1])
        else:
            tensor_2d = tensor_cpu
            
        try:
            s = torch.linalg.svdvals(tensor_2d.float())
            return s.numpy()
        except Exception as e:
            print(f"SVD computation failed: {e}")
            return np.array([])
    
    def plot_eigenvalue_distribution(self, 
                                     key: Optional[str] = None, 
                                     step: Optional[int] = None,
                                     data_type: str = 'activation',
                                     save_path: Optional[str] = None,
                                     log_scale: bool = True,
                                     top_k: Optional[int] = None,
                                     num_bins: int = 100):
        """
        Plot the spectral density (histogram) of eigenvalues for a specific key and step(s).
        Only plots steps defined in EIGENVALUES_PLOT_STEP when step is None or the specific step is requested.
        
        Args:
            key: Layer key to plot (None to plot all keys)
            step: Step index to plot (must be in EIGENVALUES_PLOT_STEP). If None, plot all steps in EIGENVALUES_PLOT_STEP.
            data_type: Type of data to plot ('activation', 'delta', or 'delta_delta')
            save_path: Path to save the plot (None to display)
            log_scale: Whether to use log scale for y-axis (density)
            top_k: Number of top eigenvalues to mention in the title (does not filter data for histogram).
            num_bins: Number of bins for the histogram.
        """
        if not self.eigenvalues:
            print("No eigenvalue data available.")
            return
            
        plt.figure(figsize=(10, 6))
        
        target_steps = EIGENVALUES_PLOT_STEPS
        target_layers = EIGENVALUES_PLOT_LAYERS
        
        if key is None:
            # Plot average density across all keys for specified steps
            all_eigenvalues_for_hist = []
            
            steps_to_consider = []
            if step is None:
                # Consider all steps defined in the global list
                steps_to_consider = target_steps
            elif step in target_steps:
                # Consider only the specified step if it's in the list
                steps_to_consider = [step]
            else:
                 print(f"Step {step} is not in EIGENVALUES_PLOT_STEP {target_steps}. Skipping plot.")
                 plt.close() # Close the figure
                 return

            num_valid_steps_found = 0
            for k in self.eigenvalues:
                if data_type in self.eigenvalues[k] and self.eigenvalues[k][data_type]:
                    for s in steps_to_consider:
                        if s < len(self.eigenvalues[k][data_type]):
                            step_eigenvalues = self.eigenvalues[k][data_type][s]
                            if len(step_eigenvalues) > 0:
                                all_eigenvalues_for_hist.extend(step_eigenvalues)
                                num_valid_steps_found +=1 # Count how many valid step arrays we added

            if not all_eigenvalues_for_hist:
                step_info = f"step(s) {steps_to_consider}" if steps_to_consider else "any valid step"
                print(f"No eigenvalue data available for {data_type} in {step_info} across all keys.")
                plt.close() # Close the figure
                return
                
            # Plot histogram for all collected eigenvalues
            all_eigenvalues_for_hist = np.concatenate(all_eigenvalues_for_hist) if isinstance(all_eigenvalues_for_hist[0], np.ndarray) else np.array(all_eigenvalues_for_hist)
            
            plt.hist(all_eigenvalues_for_hist, bins=num_bins, density=True, alpha=0.7, color='blue')
            
            title = f"Average {data_type.capitalize()} Spectral Density"
            if steps_to_consider:
                 title += f" (Step(s) {steps_to_consider})"
            if top_k is not None:
                 title += f" (Top {top_k} mentioned)" # Clarify top_k usage
            plt.title(title)
            
        else:
            # Plot for specific key
            if key not in self.eigenvalues:
                print(f"No eigenvalue data for key {key}.")
                plt.close()
                return
                
            if data_type not in self.eigenvalues[key] or not self.eigenvalues[key][data_type]:
                print(f"No {data_type} eigenvalue data for key {key}.")
                plt.close()
                return
            
            plotted_something = False
            if step is None:
                for i in target_steps:
                    if i < len(self.eigenvalues[key][data_type]):
                        step_eigenvalues = self.eigenvalues[key][data_type][i]
                        if len(step_eigenvalues) > 0:
                            plt.hist(step_eigenvalues, bins=num_bins, density=True, 
                                     alpha=0.6, label=f"Step {i}")
                            plotted_something = True
                    else:
                         print(f"Warning: Step {i} requested but out of range for key {key} ({data_type}).")

                if plotted_something and len(target_steps) > 1: # Add legend only if multiple steps could have been plotted
                    plt.legend()

                title = f"{key} {data_type.capitalize()} Spectral Density"
                if target_steps:
                     title += f" (Step(s) {target_steps})"
                if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                plt.title(title)
                
            elif step in target_steps:
                # Plot histogram for the specific step if it's in the target list
                 if step - 1 < len(self.eigenvalues[key][data_type]):
                    step_eigenvalues = self.eigenvalues[key][data_type][step - 1]
                    if len(step_eigenvalues) > 0:
                        plt.hist(step_eigenvalues, bins=num_bins, density=True, 
                                 alpha=0.7, color='red')
                        plotted_something = True
                    else:
                        print(f"No eigenvalue data for key {key}, data_type {data_type} at step {step}.")
                 else:
                      print(f"Step {step} is out of range for key {key}, data_type {data_type}.")

                 title = f"{key} {data_type.capitalize()} Spectral Density (Step {step})"
                 if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                 plt.title(title)
            else:
                 print(f"Requested Step {step} is not in EIGENVALUES_PLOT_STEP {target_steps}. Skipping plot for key {key}.")
                 plotted_something = False # Ensure we don't proceed to show/save

            if not plotted_something:
                 print(f"No data plotted for key {key}, data_type {data_type}.")
                 plt.close() # Close the figure if nothing was plotted
                 return

        plt.xlabel("Eigenvalue Magnitude")
        plt.ylabel("Spectral Density")
        
        if log_scale:
            plt.yscale('log') # Apply log scale to density axis if requested
            
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close() # Close the figure after saving
        else:
            plt.show()
            
    def plot_eigenvalue_decay(self, 
                              keys: Optional[List[str]] = None, 
                              data_types: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              log_scale: bool = True,
                              normalized: bool = True,
                              top_k: Optional[int] = None):
        """
        Plot the eigenvalue decay for multiple keys and data types.
        
        Args:
            keys: List of layer keys to plot (None for all keys)
            data_types: List of data types to plot (None for all types)
            save_path: Path to save the plot (None to display)
            log_scale: Whether to use log scale for y-axis
            normalized: Whether to normalize eigenvalues by the largest eigenvalue
            top_k: Number of top eigenvalues to plot (None for all)
        """
        if not self.eigenvalues:
            print("No eigenvalue data available.")
            return
            
        if keys is None:
            keys = list(self.eigenvalues.keys())
            
        if data_types is None:
            data_types = ['activation', 'delta', 'delta_delta']
            
        plt.figure(figsize=(12, 8))
        
        for key in keys:
            if key not in self.eigenvalues:
                continue
                
            for data_type in data_types:
                if data_type not in self.eigenvalues[key] or not self.eigenvalues[key][data_type]:
                    continue
                    
                # Average eigenvalues across all steps
                all_step_eigenvalues = []
                for step_eigenvalues in self.eigenvalues[key][data_type]:
                    if len(step_eigenvalues) > 0:
                        all_step_eigenvalues.append(step_eigenvalues)
                
                if not all_step_eigenvalues:
                    continue
                    
                # Find minimum length to align arrays
                min_length = min(len(ev) for ev in all_step_eigenvalues)
                
                # Truncate all arrays to the same length
                aligned_eigenvalues = [ev[:min_length] for ev in all_step_eigenvalues]
                
                # Compute average eigenvalues
                avg_eigenvalues = np.mean(aligned_eigenvalues, axis=0)
                
                # Apply top_k filter if specified
                if top_k is not None and top_k < len(avg_eigenvalues):
                    avg_eigenvalues = avg_eigenvalues[:top_k]
                    
                # Normalize if requested
                if normalized and len(avg_eigenvalues) > 0:
                    avg_eigenvalues = avg_eigenvalues / avg_eigenvalues[0]
                    
                plt.plot(range(1, len(avg_eigenvalues) + 1), avg_eigenvalues, 
                         linewidth=2, label=f"{key} ({data_type})")
        
        plt.xlabel("Eigenvalue Index")
        
        if normalized:
            plt.ylabel("Normalized Eigenvalue Magnitude")
        else:
            plt.ylabel("Eigenvalue Magnitude")
            
        if log_scale:
            plt.yscale('log')
            
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        
        title = "Eigenvalue Decay Comparison"
        if top_k is not None:
            title += f" (Top {top_k})"
        if normalized:
            title += " (Normalized)"
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

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

def plot_eigenvalues(key=None, step=None, data_type='activation', save_path=None, log_scale=True, top_k=None):
    """
    Global function to plot eigenvalue distribution.
    
    Args:
        key: Layer key to plot (None for average across all keys)
        step: Step index to plot (None for average across all steps)
        data_type: Type of data to plot ('activation', 'delta', or 'delta_delta')
        save_path: Path to save the plot (None to display)
        log_scale: Whether to use log scale for y-axis
        top_k: Number of top eigenvalues to plot (None for all)
    """
    if _stats is None:
        print("No statistics logged.")
        return
    _stats.plot_eigenvalue_distribution(key, step, data_type, save_path, log_scale, top_k)

def plot_eigenvalue_decay(keys=None, data_types=None, save_path=None, log_scale=True, normalized=True, top_k=None):
    """
    Global function to plot eigenvalue decay for multiple keys and data types.
    
    Args:
        keys: List of layer keys to plot (None for all keys)
        data_types: List of data types to plot (None for all types)
        save_path: Path to save the plot (None to display)
        log_scale: Whether to use log scale for y-axis
        normalized: Whether to normalize eigenvalues by the largest eigenvalue
        top_k: Number of top eigenvalues to plot (None for all)
    """
    if _stats is None:
        print("No statistics logged.")
        return
    _stats.plot_eigenvalue_decay(keys, data_types, save_path, log_scale, normalized, top_k)
