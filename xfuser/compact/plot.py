import torch
import matplotlib.pyplot as plt
import numpy as np
import os
PLOT_DIR = "plots"
from typing import Optional

def plot_3d(tensor, title, filename=None):
    # Plot
    tensor = tensor.cpu()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(tensor.shape[1]), np.arange(tensor.shape[0]))
    z = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
    ax.plot_surface(x, y, z, cmap='coolwarm', linewidth=0, antialiased=False)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Token')
    ax.set_zlabel('Tenor')
    plt.title(title)

    # Save to file
    if filename is None:
        filename = f"{PLOT_DIR}/3d_{title}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig, ax


def plot_low_rank_factors(
    u: torch.Tensor,
    v: torch.Tensor,
    key: str,
    step: Optional[int],
    save_dir: Optional[str] = None,
):
    """
    Plots the U and V factor matrices from low-rank decomposition.

    Args:
        u: The U factor matrix (N, K).
        v: The V factor matrix (K, C).
        key: The identifier for the layer/tensor.
        step: The current step index (for filename).
        save_dir: Directory to save the plot. If None, displays the plot.
    """
    if step is None:
        raise ValueError("Step is None for key {key}, cannot save U/V plot with step index.")
    else:
        step_str = f"step{step}"

    u_np = u.detach().cpu().float().numpy()
    v_np = v.detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Low-Rank Factors for {key} ({step_str})")

    # Plot U
    im_u = axes[0].imshow(u_np, aspect="auto", cmap="viridis")
    axes[0].set_title(f"U Matrix (Shape: {u_np.shape})")
    axes[0].set_xlabel("Rank (K)")
    axes[0].set_ylabel("Tokens (N)")
    fig.colorbar(im_u, ax=axes[0])

    # Plot V
    im_v = axes[1].imshow(v_np, aspect="auto", cmap="viridis")
    axes[1].set_title(f"V Matrix (Shape: {v_np.shape})")
    axes[1].set_xlabel("Channels (C)")
    axes[1].set_ylabel("Rank (K)")
    fig.colorbar(im_v, ax=axes[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{key}_{step_str}_uv.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved U/V plot to {filepath}")
        plt.close(fig)  # Close the figure after saving
    else:
        plt.show()


def plot_eigenvalue_cumsum(
    eigenvalues,
    key: Optional[str] = None,
    step: Optional[int] = None,
    data_type: str = "activation",
    save_dir: Optional[str] = None,
    log_scale: bool = True,
    top_k: Optional[int] = None,
):
    """
    Plot the cumulative distribution function (CDF) of eigenvalues for a specific key and step(s).
    Only plots steps defined in EIGENVALUES_PLOT_STEP when step is None or the specific step is requested.
    
    Args:
        key: Layer key to plot (None to plot all keys)
        step: Step index to plot (must be in EIGENVALUES_PLOT_STEP). If None, plot all steps in EIGENVALUES_PLOT_STEP.
        data_type: Type of data to plot ('activation', 'delta', or 'delta_delta')
        save_dir: Directory to save the plot (None to display)
        log_scale: Whether to use log scale for y-axis
        top_k: Number of top eigenvalues to mention in the title (does not filter data for CDF).
        num_bins: Number of bins for the histogram (used for binning before cumsum calculation).
    """
    if not eigenvalues:
        print("No eigenvalue data available.")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    gaussian_data = np.random.normal(0, 1, (2176, 3072))
    gaussian_eigenvalues = torch.linalg.svdvals(torch.from_numpy(gaussian_data))
    gaussian_eigenvalues = np.sort(gaussian_eigenvalues)[::-1]
    gaussian_cumulative = np.cumsum(gaussian_eigenvalues) / np.sum(gaussian_eigenvalues)

    if key is None and step is None: # Plot all eigenvalues for all target layers and target steps
        for key in eigenvalues:
            for step in eigenvalues[key]:
                if data_type in eigenvalues[key][step]:
                    print(f"Plotting {key} {data_type} CDF for step {step}")
                    plt.figure(figsize=(10, 6))
                    # Check if the list is empty before accessing
                    if not eigenvalues[key][step][data_type]:
                        print(f"Skipping empty eigenvalue data for {key}, step {step}, type {data_type}")
                        plt.close() # Close the empty figure
                        continue
                    eigenvalues = np.sort(eigenvalues[key][step][data_type][0])[::-1]
                    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
                    plt.plot(cumulative, label=f"Step {step}")
                    plt.plot(gaussian_cumulative, label="Gaussian distribution")
                    title = f"{key} {data_type.capitalize()} Eigenvalue CDF (Step {step})"
                    if top_k is not None:
                        title += f" (Top {top_k} mentioned)"
                    plt.title(title)
                    plt.ylabel("Cumulative Probability")
                    if log_scale:
                        plt.xscale('log')
                    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                    if save_dir:
                        file_path = os.path.join(save_dir, f"{key}_{data_type}_cdf_step{step}.png")
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        print(f"Plot saved to {file_path}")
                        plt.clf()
                        plt.close()
                    else:
                        plt.show()
    elif key is not None and step is None: # Plot all eigenvalues for all target steps for a specific layer
        if key not in eigenvalues:
            print(f"No eigenvalue data for key {key}.")
            return
        for step in eigenvalues[key]:
            if data_type in eigenvalues[key][step]:
                print(f"Plotting {key} {data_type} CDF for step {step}")
                plt.figure(figsize=(10, 6))

                # Check if the list is empty before accessing
                if not eigenvalues[key][step][data_type]:
                    print(f"Skipping empty eigenvalue data for {key}, step {step}, type {data_type}")
                    plt.close() # Close the empty figure
                    continue
                # Sort eigenvalues and calculate cumulative distribution
                eigenvalues = np.sort(eigenvalues[key][step][data_type][0])[::-1]
                cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)

                plt.plot(cumulative, label=f"Step {step}")
                plt.plot(gaussian_cumulative, label="Gaussian distribution")

                title = f"{key} {data_type.capitalize()} Eigenvalue CDF (Step {step})"
                if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                plt.title(title)
                plt.ylabel("Cumulative Probability")
                if log_scale:
                    plt.xscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                if save_dir:
                    file_path = os.path.join(save_dir, f"{key}_{data_type}_cdf_step{step}.png")
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {file_path}")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()
    elif key is None and step is not None: # Plot all eigenvalues for all layers for a specific step
        for key in eigenvalues:
            if step not in eigenvalues[key]:
                continue
            if data_type in eigenvalues[key][step]:
                print(f"Plotting {key} {data_type} CDF for step {step}")
                plt.figure(figsize=(10, 6))

                # Check if the list is empty before accessing
                if not eigenvalues[key][step][data_type]:
                    print(f"Skipping empty eigenvalue data for {key}, step {step}, type {data_type}")
                    plt.close() # Close the empty figure
                    continue
                # Sort eigenvalues and calculate cumulative distribution
                eigenvalues = np.sort(eigenvalues[key][step][data_type][0])[::-1]
                cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)

                plt.plot(cumulative, label=f"Step {step}")
                plt.plot(gaussian_cumulative, label="Gaussian distribution")

                title = f"{key} {data_type.capitalize()} Eigenvalue CDF (Step {step})"
                if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                plt.title(title)
                plt.ylabel("Cumulative Probability")
                if log_scale:
                    plt.xscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                if save_dir:
                    file_path = os.path.join(save_dir, f"{key}_{data_type}_cdf_step{step}.png")
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {file_path}")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()
    elif key is not None and step is not None: # Plot eigenvalues for a specific key and step
        if key not in eigenvalues:
            print(f"No eigenvalue data for key {key}.")
            return

        if step not in eigenvalues[key]:
            print(f"No eigenvalue data for key {key} and step {step}.")
            return

        if data_type not in eigenvalues[key][step] or not eigenvalues[key][step][data_type]:
            print(f"No {data_type} eigenvalue data for key {key} and step {step}.")
            return

        print(f"Plotting {key} {data_type} CDF for step {step}")
        plt.figure(figsize=(10, 6))

        # Check if the list is empty before accessing
        if not eigenvalues[key][step][data_type]:
            print(f"Skipping empty eigenvalue data for {key}, step {step}, type {data_type}")
            plt.close() # Close the empty figure
            return # Exit the function if data is missing for the specific key/step
        # Sort eigenvalues and calculate cumulative distribution
        eigenvalues = np.sort(eigenvalues[key][step][data_type][0])[::-1]
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        plt.plot(cumulative, label=f"Step {step}")
        plt.plot(gaussian_cumulative, label="Gaussian distribution")

        title = f"{key} {data_type.capitalize()} Eigenvalue CDF (Step {step})"
        if top_k is not None:
            title += f" (Top {top_k} mentioned)"
        plt.title(title)
        plt.ylabel("Cumulative Probability")
        if log_scale:
            plt.xscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_dir:
            file_path = os.path.join(save_dir, f"{key}_{data_type}_cdf_step{step}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {file_path}")
            plt.clf()
            plt.close()
        else:
            plt.show()


def plot_eigenvalue_distribution(
    eigenvalues,
    key: Optional[str] = None,
    step: Optional[int] = None,
    data_type: str = "activation",
    save_dir: Optional[str] = None,
    log_scale: bool = True,
    top_k: Optional[int] = None,
    num_bins: int = 100,
):
    """
    Plot the spectral density (histogram) of eigenvalues for a specific key and step(s).
    Only plots steps defined in EIGENVALUES_PLOT_STEP when step is None or the specific step is requested.
    
    Args:
        key: Layer key to plot (None to plot all keys)
        step: Step index to plot (must be in EIGENVALUES_PLOT_STEP). If None, plot all steps in EIGENVALUES_PLOT_STEP.
        data_type: Type of data to plot ('activation', 'delta', or 'delta_delta')
        save_dir: Directory to save the plot (None to display)
        log_scale: Whether to use log scale for y-axis (density)
        top_k: Number of top eigenvalues to mention in the title (does not filter data for histogram).
        num_bins: Number of bins for the histogram.
    """
    if not eigenvalues:
        print("No eigenvalue data available.")
        return

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if key is None and step is None: # Plot all eigenvalues for all target layers and target steps
        for key in eigenvalues:
            for step in eigenvalues[key]:
                if data_type in eigenvalues[key][step]:
                    print(f"Plotting {key} {data_type} for step {step}")
                    plt.figure(figsize=(10, 6))
                    plt.hist(eigenvalues[key][step][data_type], bins=num_bins, density=True, 
                            alpha=0.6, label=f"Step {step}")
                    title = f"{key} {data_type.capitalize()} Spectral Density (Step {step})"
                    if top_k is not None:
                        title += f" (Top {top_k} mentioned)"
                    plt.title(title)
                    plt.xlabel("Eigenvalue Magnitude")
                    plt.ylabel("Spectral Density")
                    if log_scale:
                        plt.yscale('log')
                    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                    if save_dir:
                        file_path = os.path.join(save_dir, f"{key}_{data_type}_step{step}.png")
                        plt.savefig(file_path, dpi=300, bbox_inches='tight')
                        print(f"Plot saved to {file_path}")
                        plt.clf()
                        plt.close()
                    else:
                        plt.show()
    elif key is not None and step is None: # Plot all eigenvalues for all target steps for a specific layer
        if key not in eigenvalues:
            print(f"No eigenvalue data for key {key}.")
            return
        for step in eigenvalues[key]:
            if data_type in eigenvalues[key][step]:
                print(f"Plotting {key} {data_type} for step {step}")
                plt.figure(figsize=(10, 6))
                plt.hist(eigenvalues[key][step][data_type], bins=num_bins, density=True, 
                        alpha=0.6, label=f"Step {step}")
                title = f"{key} {data_type.capitalize()} Spectral Density (Step {step})"
                if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                plt.title(title)
                plt.xlabel("Eigenvalue Magnitude")
                plt.ylabel("Spectral Density")
                if log_scale:
                    plt.yscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                if save_dir:
                    file_path = os.path.join(save_dir, f"{key}_{data_type}_step{step}.png")
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {file_path}")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()
    elif key is None and step is not None: # Plot all eigenvalues for all layers for a specific step
        for key in eigenvalues:
            if step not in eigenvalues[key]:
                continue
            if data_type in eigenvalues[key][step]:
                print(f"Plotting {key} {data_type} for step {step}")
                plt.figure(figsize=(10, 6))
                plt.hist(eigenvalues[key][step][data_type], bins=num_bins, density=True, 
                        alpha=0.6, label=f"Step {step}")
                title = f"{key} {data_type.capitalize()} Spectral Density (Step {step})"
                if top_k is not None:
                    title += f" (Top {top_k} mentioned)"
                plt.title(title)
                plt.xlabel("Eigenvalue Magnitude")
                plt.ylabel("Spectral Density")
                if log_scale:
                    plt.yscale('log')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                if save_dir:
                    file_path = os.path.join(save_dir, f"{key}_{data_type}_step{step}.png")
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"Plot saved to {file_path}")
                    plt.clf()
                    plt.close()
                else:
                    plt.show()
    elif key is not None and step is not None: # Plot eigenvalues for a specific key and step
        if key not in eigenvalues:
            print(f"No eigenvalue data for key {key}.")
            return

        if step not in eigenvalues[key]:
            print(f"No eigenvalue data for key {key} and step {step}.")
            return

        if data_type not in eigenvalues[key][step] or not eigenvalues[key][step][data_type]:
            print(f"No {data_type} eigenvalue data for key {key} and step {step}.")
            return

        print(f"Plotting {key} {data_type} for step {step}")
        plt.figure(figsize=(10, 6))
        plt.hist(eigenvalues[key][step][data_type], bins=num_bins, density=True, 
                alpha=0.6, label=f"Step {step}")
        title = f"{key} {data_type.capitalize()} Spectral Density (Step {step})"
        if top_k is not None:
            title += f" (Top {top_k} mentioned)"
        plt.title(title)
        plt.xlabel("Eigenvalue Magnitude")
        plt.ylabel("Spectral Density")
        if log_scale:
            plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if save_dir:
            file_path = os.path.join(save_dir, f"{key}_{data_type}_step{step}.png")
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {file_path}")
            plt.clf()
            plt.close()
        else:
            plt.show()


def dump_average_error_vs_steps(
    stats_data,
    save_dir: str, # Made save_dir mandatory for dumping
):
    """
    Calculates average compression error and average total error vs steps across all keys
    and dumps the data to a file.

    Args:
        stats_data: The dictionary containing the statistics (e.g., _stats.stats).
        save_dir: Directory to save the dumped data file.
    """
    if not stats_data:
        print("Error: No statistics data provided. Cannot dump error data.")
        return

    # Determine the maximum number of steps recorded across all keys
    max_steps = 0
    for key in stats_data:
        max_steps = max(max_steps, len(stats_data[key]))

    if max_steps == 0:
        print("Error: No steps logged in statistics data. Cannot dump error data.")
        return

    avg_comp_errors = []
    avg_total_errors = []
    steps = list(range(max_steps)) # Ensure steps is a list for saving

    for step in steps:
        step_comp_errors = []
        step_total_errors = []
        has_total_error_this_step = False
        for key in stats_data:
            if step < len(stats_data[key]):
                stat = stats_data[key][step]
                step_comp_errors.append(stat['error'])
                if stat.get('total_error') is not None:
                    step_total_errors.append(stat['total_error'])
                    has_total_error_this_step = True

        # Calculate average for the current step, handle cases with no data
        avg_comp_errors.append(float(np.mean(step_comp_errors)) if step_comp_errors else None) # Use None for missing data
        avg_total_errors.append(float(np.mean(step_total_errors)) if has_total_error_this_step and step_total_errors else None) # Use None if no total err this step

    print("Dumped avg error data:")
    print('steps:', steps)
    print('avg_comp_errors:', [f"{x:.2f}" if x is not None else None for x in avg_comp_errors])
    print('avg_total_errors:', [f"{x:.2f}" if x is not None else None for x in avg_total_errors])


    # Prepare original data for saving (without formatting)
    dump_data = {
        'steps': steps,
        'avg_comp_errors': avg_comp_errors,
        'avg_total_errors': avg_total_errors,
    }
    # Save the original (unformatted) data
    os.makedirs(save_dir, exist_ok=True)
    filename = "average_error_vs_steps.pt"
    filepath = os.path.join(save_dir, filename)
    try:
        torch.save(dump_data, filepath)
        print(f"Saved average error data to {filepath}")
    except Exception as e:
        print(f"Error saving average error data to {filepath}: {e}")


def dump_average_norms_and_similarity_vs_steps(
    stats_data,
    save_dir: str, # Made save_dir mandatory
):
    """
    Calculates average activation norm, delta norm, and activation similarity vs steps across all keys
    and dumps the data to a file.

    Args:
        stats_data: The dictionary containing the statistics (e.g., _stats.stats).
        save_dir: Directory to save the dumped data file.
    """
    if not stats_data:
        print("Error: No statistics data provided. Cannot dump norms/similarity data.")
        return

    # Determine the maximum number of steps recorded across all keys
    max_steps = 0
    for key in stats_data:
        max_steps = max(max_steps, len(stats_data[key]))

    if max_steps == 0:
        print("Error: No steps logged in statistics data. Cannot dump norms/similarity data.")
        return

    avg_act_norms = []
    avg_delta_norms = []
    avg_act_similarities = []
    steps = list(range(max_steps))

    for step in steps:
        step_act_norms = []
        step_delta_norms = []
        step_act_similarities = []

        for key in stats_data:
            if step < len(stats_data[key]):
                stat = stats_data[key][step]
                # Activation Norm (always present)
                step_act_norms.append(stat['activation_norm'])

                # Delta Norm (present if residual >= 1)
                if stat['delta_norm'] is not None:
                    step_delta_norms.append(stat['delta_norm'])

                # Activation Similarity (present if calculated)
                if stat['activation_similarity'] is not None:
                    step_act_similarities.append(stat['activation_similarity'])

        # Calculate averages for the current step, handle cases with no data
        avg_act_norms.append(float(np.mean(step_act_norms)) if step_act_norms else None)
        avg_delta_norms.append(float(np.mean(step_delta_norms)) if step_delta_norms else None)
        avg_act_similarities.append(float(np.mean(step_act_similarities)) if step_act_similarities else None)

    # Print formatted data for viewing
    print("Dumped avg norms and similarity data:")
    print('steps:', steps)
    print('avg_act_norms:', [f"{x:.3f}" if x is not None else None for x in avg_act_norms])
    print('avg_delta_norms:', [f"{x:.3f}" if x is not None else None for x in avg_delta_norms])
    print('avg_act_similarities:', [f"{x:.3f}" if x is not None else None for x in avg_act_similarities])

    # Prepare original data for saving (without formatting)
    dump_data = {
        'steps': steps,
        'avg_act_norms': avg_act_norms,
        'avg_delta_norms': avg_delta_norms,
        'avg_act_similarities': avg_act_similarities,
    }

    # Save the original (unformatted) data
    os.makedirs(save_dir, exist_ok=True)
    filename = "average_norms_and_similarity_vs_steps.pt"
    filepath = os.path.join(save_dir, filename)
    try:
        torch.save(dump_data, filepath)
        print(f"Saved average norms and similarity data to {filepath}")
    except Exception as e:
        print(f"Error saving average norms and similarity data to {filepath}: {e}")
