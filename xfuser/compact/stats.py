import torch
import numpy as np
import random
import os # Added for path operations
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

# EIGENVALUES_PLOT_STEPS = [10]
# EIGENVALUES_PLOT_LAYERS = [10, 20]
EIGENVALUES_PLOT_STEPS = []
EIGENVALUES_PLOT_LAYERS = []

# UV_PLOT_STEPS = [10]
# UV_PLOT_LAYERS = [10, 20]
UV_PLOT_STEPS = []
UV_PLOT_LAYERS = []

# ENV VARS
CALC_SIMILARITY = os.environ.get("CALC_SIMILARITY", "0") == "1"
CALC_MORE_SIMILARITY = os.environ.get("CALC_MORE_SIMILARITY", "0") == "1"
PRINT_ALL_ERROR = os.environ.get("PRINT_ALL_ERROR", "0") == "1"

REF_ACTIVATION_PATH = os.environ.get("REF_ACTIVATION_PATH", "ref_activations")
DUMP_ACTIVATIONS = os.environ.get("DUMP_ACTIVATIONS", "0") == "1"
CALC_TOTAL_ERROR = os.environ.get("CALC_TOTAL_ERROR", "0") == "1"

def stats_hello():
    print("--- 📦  Statistics Configuration ---")
    print(f"CALC_SIMILARITY: {CALC_SIMILARITY}")
    print(f"CALC_MORE_SIMILARITY: {CALC_MORE_SIMILARITY}")
    print(f"PRINT_ALL_ERROR: {PRINT_ALL_ERROR}")
    print(f"REF_ACTIVATION_PATH: {REF_ACTIVATION_PATH}")
    print(f"DUMP_ACTIVATIONS: {DUMP_ACTIVATIONS}")
    print(f"CALC_TOTAL_ERROR: {CALC_TOTAL_ERROR}")
    print("------------------------------")

class StatsLogger:
    """Simple statistics logger for compression metrics."""

    def __init__(self):
        # Main storage for stats
        self.stats = {}
        # Prev step storage for similarity calculations
        self.prev_activations = {}
        self.prev_deltas = {}
        self.prev_transmitted_deltas = {} # Added storage for transmitted delta
        self.prev_delta_before_feedback = {} # Added storage for delta before feedback
        self.prev_delta_before_feedback_lowrank = {} # Added storage for low rank delta before feedback
        self.plot_counter = 0 # for plotting
        # For volume tracking
        self.total_original_volume = 0
        self.total_compressed_volume = 0
        # Track steps per key for filenames
        self.step_counts = {} 
        # Store eigenvalues for analysis
        self.eigenvalues = {}

    def _compute_strided_row_similarity(self, tensor: torch.Tensor, stride: int = 1) -> Optional[float]:
        """
        Computes the average cosine similarity between rows separated by a stride
        using vectorized operations.

        Args:
            tensor: A 2D tensor (N, C).
            stride: The step size between rows to compare (default: 1 for adjacent).

        Returns:
            The average similarity, or None if N <= stride or no valid pairs found.
        """
        assert tensor is not None and tensor.ndim == 2, f"Invalid tensor input for strided row similarity: ndim={tensor.ndim if tensor is not None else 'None'}"
        assert tensor.shape[0] > stride, f"Cannot compute similarity if not enough rows for the stride (stride={stride}). Shape: {tensor.shape}"

        N, C = tensor.shape

        # Check for non-finite values in the entire tensor first
        assert torch.isfinite(tensor).all(), f"Non-finite values found in tensor for strided row similarity (stride={stride}). Shape: {tensor.shape}"

        rows1 = tensor[:-stride, :] # Shape (N-stride, C)
        rows2 = tensor[stride:, :]  # Shape (N-stride, C)

        # Calculate norms for all relevant rows efficiently
        norms1 = torch.linalg.norm(rows1, dim=1) # Shape (N-stride)
        norms2 = torch.linalg.norm(rows2, dim=1) # Shape (N-stride)

        # Identify pairs where either row has near-zero norm
        valid_mask = (norms1 > 1e-8) & (norms2 > 1e-8) # Shape (N-stride)

        # Handle case where no pairs are valid
        assert valid_mask.any(), f"No valid strided row pairs found for similarity calculation (stride={stride}). Shape: {tensor.shape}"

        # Select only valid rows to compute similarity
        valid_rows1 = rows1[valid_mask]
        valid_rows2 = rows2[valid_mask]

        # Compute cosine similarity for valid pairs
        similarities = torch.nn.functional.cosine_similarity(valid_rows1, valid_rows2, dim=1, eps=1e-8) # Shape (num_valid_pairs)

        assert similarities.numel() > 0, f"Similarities tensor became empty after filtering (stride={stride}). Shape: {tensor.shape}"

        avg_similarity = torch.mean(similarities)

        # Final check for NaN just in case
        assert not torch.isnan(avg_similarity), f"Average strided row similarity resulted in NaN (stride={stride}). Shape: {tensor.shape}"

        return avg_similarity.item()

    def log(
        self,
        key, 
        base, 
        delta_base, 
        before_comp_activation, 
        recv_activation, 
        compressed_tensor, 
        compress_residual,
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
        """

        ref_activation_path = REF_ACTIVATION_PATH
        dump_activations = DUMP_ACTIVATIONS
        calc_total_error = CALC_TOTAL_ERROR
        
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
        delta = None
        delta_delta = None
        transmitted_delta = None
        transmitted_delta_sim = None
        
        # Calculate delta before feedback (diff with actual previous activation, not delta_base)
        delta_before_feedback = None
        delta_before_feedback_norm = None
        delta_before_feedback_sim = None
        delta_before_feedback_lowrank = None
        delta_before_feedback_lowrank_sim = None  # Added similarity for low rank version
        
        if key in self.prev_activations:
            delta_before_feedback = before_comp_activation - self.prev_activations[key].to(before_comp_activation.device)
            delta_before_feedback_norm = torch.norm(delta_before_feedback).item()
            
            from xfuser.compact.slowpath import sim_compress
            from xfuser.compact.utils import COMPACT_COMPRESS_TYPE
            delta_before_feedback_lowrank = sim_compress(delta_before_feedback.cuda(), COMPACT_COMPRESS_TYPE.LOW_RANK, rank=8).cpu()
            # no need for norm, just for similarity

        if compress_residual == 0:
            pass # No deltas calculated
        elif compress_residual == 1:
            delta = before_comp_activation - base
            transmitted_delta = recv_activation - base
            delta_delta = None
        elif compress_residual == 2:
            delta = before_comp_activation - base
            transmitted_delta = recv_activation - base # Still recv_activation - base
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

        if CALC_SIMILARITY:
            # Previous step similarities
            if key in self.prev_activations:
                # Ensure tensors are flat and on the same device for similarity
                act_sim = torch.nn.functional.cosine_similarity(
                    before_comp_activation.flatten(),
                    self.prev_activations[key].flatten().to(before_comp_activation.device),
                    dim=0,
                    eps=1e-8 # Add epsilon for numerical stability
                ).item()

            if delta is not None and key in self.prev_deltas:
                current_delta_flat = delta.flatten()
                prev_delta_flat = self.prev_deltas[key].flatten().to(delta.device)
                current_norm = torch.norm(current_delta_flat)
                prev_norm = torch.norm(prev_delta_flat)
                delta_sim = torch.nn.functional.cosine_similarity(
                    current_delta_flat, # Use flattened tensors
                    prev_delta_flat,
                    dim=0,
                    eps=1e-8 # Add epsilon for numerical stability
                ).item()
            
            if CALC_MORE_SIMILARITY:
                # Calculate transmitted delta similarity
                if transmitted_delta is not None and key in self.prev_transmitted_deltas:
                    current_transmitted_delta_flat = transmitted_delta.flatten()
                    prev_transmitted_delta_flat = self.prev_transmitted_deltas[key].flatten().to(transmitted_delta.device)
                    current_norm = torch.norm(current_transmitted_delta_flat)
                    prev_norm = torch.norm(prev_transmitted_delta_flat)
                    transmitted_delta_sim = torch.nn.functional.cosine_similarity(
                        current_transmitted_delta_flat,
                        prev_transmitted_delta_flat,
                        dim=0,
                        eps=1e-8 # Add epsilon for numerical stability
                    ).item()
                # Calculate delta before feedback similarity
                if delta_before_feedback is not None and key in self.prev_delta_before_feedback:
                    current_dbf_flat = delta_before_feedback.flatten()
                    prev_dbf_flat = self.prev_delta_before_feedback[key].flatten().to(delta_before_feedback.device)
                    delta_before_feedback_sim = torch.nn.functional.cosine_similarity(
                        current_dbf_flat,
                        prev_dbf_flat,
                        dim=0,
                        eps=1e-8 # Add epsilon for numerical stability
                    ).item()
                    
                # Calculate low rank delta before feedback similarity
                if delta_before_feedback_lowrank is not None and key in self.prev_delta_before_feedback_lowrank:
                    current_dbf_lr_flat = delta_before_feedback_lowrank.flatten()
                    prev_dbf_lr_flat = self.prev_delta_before_feedback_lowrank[key].flatten().to(delta_before_feedback_lowrank.device)
                    delta_before_feedback_lowrank_sim = torch.nn.functional.cosine_similarity(
                        current_dbf_lr_flat,
                        prev_dbf_lr_flat,
                        dim=0,
                        eps=1e-8 # Add epsilon for numerical stability
                    ).item()
                
        # Compute Eigenvalues and Store Them
        layer_idx = int(key.split('-')[0])
        step_idx = self.step_counts[key]
        if step_idx in EIGENVALUES_PLOT_STEPS and layer_idx in EIGENVALUES_PLOT_LAYERS:
            if key not in self.eigenvalues:
                self.eigenvalues[key] = {}

            if step_idx not in self.eigenvalues[key]:
                self.eigenvalues[key][step_idx] = {'activation': [], 'delta': [], 'delta_delta': []}

            act_eigenvalues = self._compute_eigenvalues(before_comp_activation)
            self.eigenvalues[key][step_idx]['activation'].append(act_eigenvalues)

            if delta is not None:
                delta_eigenvalues = self._compute_eigenvalues(delta)
                self.eigenvalues[key][step_idx]['delta'].append(delta_eigenvalues)

            if delta_delta is not None:
                delta_delta_eigenvalues = self._compute_eigenvalues(delta_delta)
                self.eigenvalues[key][step_idx]['delta_delta'].append(delta_delta_eigenvalues)

        # Store current stats
        self.stats[key].append({
            'error': error.item(),
            'total_error': total_error.item() if total_error is not None else None, # Store total error
            'activation_norm': act_norm,
            'delta_norm': delta_norm,
            'delta_delta_norm': delta_delta_norm,
            'delta_before_feedback_norm': delta_before_feedback_norm, # Add delta before feedback norm
            'activation_similarity': act_sim,
            'delta_similarity': delta_sim,
            'delta_before_feedback_similarity': delta_before_feedback_sim, # Add delta before feedback similarity
            'delta_before_feedback_lowrank_similarity': delta_before_feedback_lowrank_sim, # Add low rank similarity
            'transmitted_delta_similarity': transmitted_delta_sim, # Added transmitted delta similarity
            'residual': compress_residual,
            'original_size_bytes': original_size_bytes,
            'compressed_size_bytes': compressed_size_bytes,
        })

        # Store current activations and deltas for next step similarity (on CPU to save GPU memory)
        self.prev_activations[key] = before_comp_activation.detach().cpu()
        if delta is not None:
            self.prev_deltas[key] = delta.detach().cpu()
        if transmitted_delta is not None: # Store transmitted delta
            self.prev_transmitted_deltas[key] = transmitted_delta.detach().cpu()
        if delta_before_feedback is not None: # Store delta before feedback
            self.prev_delta_before_feedback[key] = delta_before_feedback.detach().cpu()
        if delta_before_feedback_lowrank is not None: # Store low rank delta before feedback
            self.prev_delta_before_feedback_lowrank[key] = delta_before_feedback_lowrank.detach().cpu()

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
        s = torch.linalg.svdvals(tensor_2d.float())
        return s.numpy()

    def plot_eigenvalue_distribution(self, 
                                     key: Optional[str] = None, 
                                     step: Optional[int] = None,
                                     data_type: str = 'activation',
                                     save_dir: Optional[str] = None,
                                     log_scale: bool = True,
                                     top_k: Optional[int] = None,
                                     num_bins: int = 100):
        from xfuser.compact.plot import plot_eigenvalue_distribution
        plot_eigenvalue_distribution(self.eigenvalues, key, step, data_type, save_dir, log_scale, top_k, num_bins)

    def plot_eigenvalue_cumsum(
        self,
        key: Optional[str] = None,
        step: Optional[int] = None,
        data_type: str = "activation",
        save_dir: Optional[str] = None,
        log_scale: bool = True,
        top_k: Optional[int] = None,
    ):
        from xfuser.compact.plot import plot_eigenvalue_cumsum
        plot_eigenvalue_cumsum(self.eigenvalues, key, step, data_type, save_dir, log_scale, top_k)

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
                # print ALL error
                if PRINT_ALL_ERROR:
                    all_error = [s['error'] for s in stats]
                    print(f"all error: {all_error}")
                
                # Compute averages
                avg_error = np.mean([s['error'] for s in stats])
                avg_total_error_list = [s['total_error'] for s in stats if s['total_error'] is not None]
                avg_total_error = np.mean(avg_total_error_list) if avg_total_error_list else None
                avg_act_norm = np.mean([s['activation_norm'] for s in stats])

                avg_rel_error = avg_error / avg_act_norm if avg_act_norm > 1e-8 else float('inf')
                print(f"err: {avg_error:.3f}, rel_err: {avg_rel_error:.1%}" + (f", total_err: {avg_total_error:.3f}" if avg_total_error is not None else ""), end="")
                print(f", act: {avg_act_norm:.3f}", end="")

                if res >= 1:
                    avg_delta_norm = np.mean([s['delta_norm'] for s in stats if s['delta_norm'] is not None])
                    delta_ratio = avg_delta_norm/avg_act_norm
                    print(f", delta={avg_delta_norm:.3f}, d/a={delta_ratio:.2f}", end="")

                    # Add delta before feedback stats 
                    dbf_norm_values = [s['delta_before_feedback_norm'] for s in stats if s.get('delta_before_feedback_norm') is not None]
                    if dbf_norm_values:
                        avg_dbf_norm = np.mean(dbf_norm_values)
                        dbf_ratio = avg_dbf_norm/avg_act_norm if avg_act_norm > 1e-8 else float('inf')
                        print(f", dbf={avg_dbf_norm:.3f}, dbf/a={dbf_ratio:.2f}", end="")

                if res >= 2:
                    avg_delta_delta_norm = np.mean([s['delta_delta_norm'] for s in stats if s['delta_delta_norm'] is not None])
                    if avg_delta_norm > 0:
                        dd_ratio = avg_delta_delta_norm/avg_delta_norm
                        print(f", dd={avg_delta_delta_norm:.3f}, dd/d={dd_ratio:.2f}", end="")

                print() # Newline after norms

                # --- Helper function for adding average stats to lists ---
                def _add_avg_stat(stats_list, stat_key, label, target_list):
                    values = [s[stat_key] for s in stats_list if s.get(stat_key) is not None]
                    if values:
                        avg_value = np.mean(values)
                        target_list.append(f"{label}: {avg_value:.3f}")
                # --- End Helper ---

                # Calculate and format previous step similarities
                similarities = []
                _add_avg_stat(stats, 'activation_similarity', 'act_sim', similarities)
                if res >= 1:
                    _add_avg_stat(stats, 'delta_similarity', 'delta_sim', similarities)
                    _add_avg_stat(stats, 'transmitted_delta_similarity', 'tx_delta_sim', similarities)
                
                # Add delta before feedback similarity
                _add_avg_stat(stats, 'delta_before_feedback_similarity', 'dbf_sim', similarities)
                _add_avg_stat(stats, 'delta_before_feedback_lowrank_similarity', 'dbf_lr_sim', similarities)

                if similarities:
                    print("  " + ", ".join(similarities)) # Indent similarity line

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
        
        # Calculate average delta before feedback norm
        dbf_values = []
        for stats_list in self.stats.values():
            for s in stats_list:
                if s.get('delta_before_feedback_norm') is not None:
                    dbf_values.append(s['delta_before_feedback_norm'])
        mean_dbf = np.mean(dbf_values) if dbf_values else None

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
              (f", avg dbf: {mean_dbf:.3f}" if mean_dbf is not None else "") +
              (f", avg delta-delta: {mean_dd:.3f}" if mean_dd is not None else ""))
              
        # Helper function to calculate average of a similarity metric
        def calc_avg_similarity(metric_key):
            values = []
            for stats_list in self.stats.values():
                for s in stats_list:
                    if s.get(metric_key) is not None:
                        values.append(s[metric_key])
            return np.mean(values) if values else None
        
        # Calculate average similarities
        mean_act_sim = calc_avg_similarity('activation_similarity')
        mean_delta_sim = calc_avg_similarity('delta_similarity')
        mean_dbf_sim = calc_avg_similarity('delta_before_feedback_similarity')
        mean_dbf_lr_sim = calc_avg_similarity('delta_before_feedback_lowrank_similarity')
        mean_tx_delta_sim = calc_avg_similarity('transmitted_delta_similarity')
        
        # Print average similarities
        sim_parts = []
        if mean_act_sim is not None:
            sim_parts.append(f"act_sim: {mean_act_sim:.3f}")
        if mean_delta_sim is not None:
            sim_parts.append(f"delta_sim: {mean_delta_sim:.3f}")
        if mean_dbf_sim is not None:
            sim_parts.append(f"dbf_sim: {mean_dbf_sim:.3f}")
        if mean_dbf_lr_sim is not None:
            sim_parts.append(f"dbf_lr_sim: {mean_dbf_lr_sim:.3f}")
        if mean_tx_delta_sim is not None:
            sim_parts.append(f"tx_delta_sim: {mean_tx_delta_sim:.3f}")
            
        if sim_parts:
            print(f"🟪 avg similarities: {', '.join(sim_parts)}")

        mean_err = np.mean([np.mean([s['error'] for s in stats]) for stats in self.stats.values()])

        # Calculate average total error if available
        total_err_values = []
        for stats_list in self.stats.values():
            for s in stats_list:
                if s['total_error'] is not None:
                    total_err_values.append(s['total_error'])
        mean_total_err = np.mean(total_err_values) if total_err_values else None

        mean_rel_err = mean_err / mean_act if mean_act > 1e-8 else float('inf')
        print(f"🟧 avg comp error: {mean_err:.3f}, avg rel err: {mean_rel_err:.1%}" + (f", avg total err: {mean_total_err:.3f}" if mean_total_err is not None else ", [total err not logged]"))
        from xfuser.compact.utils import get_emoji
        print(get_emoji())

    def save_eigenvalues(self, save_dir="eigenvalues"):
        """
        Save profiled eigenvalues to a .pt file for each layer and each step.
        """
        if not self.eigenvalues:
            print("No eigenvalue data available.")
            return

        # Create a directory for saving eigenvalues
        os.makedirs(save_dir, exist_ok=True)

        # Iterate through each layer and each step
        for key, step_data in self.eigenvalues.items():
            for step, data_types in step_data.items():
                for data_type, eigenvalues in data_types.items():
                    # Create a filename for the current layer and step
                    filename = f"{key}_{step}_{data_type}.pt"
                    filepath = os.path.join(save_dir, filename)

                    # Save the eigenvalues as a PyTorch tensor
                    torch.save(eigenvalues, filepath)

        print(f"Saved eigenvalues to {save_dir}")

    def plot_low_rank_factors(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        key: str,
        step: int,
        save_dir: str,
    ):
        assert step is not None, "Step is None for key {key}, cannot save U/V plot with step index."
        layer_idx = int(key.split("-")[0])
        if layer_idx not in UV_PLOT_LAYERS or step not in UV_PLOT_STEPS:
            return
        from xfuser.compact.plot import plot_low_rank_factors
        plot_low_rank_factors(u, v, key, step, save_dir)

    def dump_average_error_vs_steps(
        self,
        save_dir: str, # Made save_dir mandatory
    ):
        """Dumps average error vs steps data to a file."""
        assert self.stats, "No statistics logged. Cannot dump data."
        # Basic check if any data exists, detailed checks happen in the dump function

        from xfuser.compact.plot import dump_average_error_vs_steps # Import renamed function
        dump_average_error_vs_steps(self.stats, save_dir) # Call renamed function

    def dump_average_norms_and_similarity_vs_steps(
        self,
        save_dir: str,
    ):
        """Dumps average activation norm, delta norm, and activation similarity vs steps data to a file."""
        assert self.stats, "No statistics logged. Cannot dump data."

        from xfuser.compact.plot import dump_average_norms_and_similarity_vs_steps
        dump_average_norms_and_similarity_vs_steps(self.stats, save_dir)

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

def plot_eigenvalues(key=None, step=None, data_type='activation', save_dir=None, log_scale=True, top_k=None, cum_sum=False):
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
    if cum_sum:
        _stats.plot_eigenvalue_cumsum(key, step, data_type, save_dir, log_scale, top_k)
    else:
        _stats.plot_eigenvalue_distribution(key, step, data_type, save_dir, log_scale, top_k)

def save_eigenvalues(save_dir="eigenvalues"):
    """
    Global function to save profiled eigenvalues.
    """
    if _stats is None:
        print("No statistics logged.")
        return
    
    _stats.save_eigenvalues(save_dir)

def dump_err_vs_steps(save_dir: str): # Renamed, save_dir mandatory
    """
    Global function to dump average compression and total error vs steps data.
    
    Args:
        save_dir: Directory to save the dumped data file.
    """
    if _stats is None:
        print("No statistics logged. Cannot dump data.")
        return
    _stats.dump_average_error_vs_steps(save_dir) # Call renamed method

def dump_norms_sim_vs_steps(save_dir: str):
    """
    Global function to dump average activation norm, delta norm, 
    and activation similarity vs steps data.
    
    Args:
        save_dir: Directory to save the dumped data file.
    """
    if _stats is None:
        print("No statistics logged. Cannot dump data.")
        return
    _stats.dump_average_norms_and_similarity_vs_steps(save_dir)