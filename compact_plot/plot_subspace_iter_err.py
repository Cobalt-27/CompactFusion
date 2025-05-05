import torch
import numpy as np
import os
from pathlib import Path
from xfuser.compact.compress_lowrank import svd, subspace_iter

def calculate_approx_error(a, b):
    """Calculate the Frobenius norm of the difference between two tensors."""
    return torch.norm(a - b).item()

def main():
    # Set up parameters
    RANKS = [8, 16]  # Different ranks to try
    ITERS = [1, 2, 3, 5, 10, 20]  # Different numbers of iterations for subspace_iter
    
    # Create output directory
    output_dir = Path("compact_plot/results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dumped tensor
    dump_path = "compact_plot/activation_dump/20-0-k_step10.pt"
    try:
        activation = torch.load(dump_path).cuda()
        print(f"Loaded tensor with shape: {activation.shape}")
    except FileNotFoundError:
        print(f"Error: File {dump_path} not found. Please check the path.")
        return
    
    # Results storage
    results = {}
    
    # For each rank, calculate errors
    for rank in RANKS:
        print(f"Processing rank {rank}...")
        results[rank] = {
            'svd_norm': 0,
            'subspace': {}
        }
        
        # Calculate optimal SVD approximation
        u_svd, v_svd = svd(activation, rank)
        svd_approx = u_svd @ v_svd
        svd_norm = torch.norm(svd_approx).item()
        
        # Store SVD norm
        results[rank]['svd_norm'] = svd_norm
        
        # Calculate subspace iteration approximation with different iterations
        for num_iter in ITERS:
            u_sub, v_sub, _ = subspace_iter(activation, rank, num_iters=num_iter)
            sub_approx = u_sub @ v_sub
            
            # Calculate error between subspace result and SVD result
            sub_vs_svd_error = calculate_approx_error(sub_approx, svd_approx)
            
            # Calculate relative error compared to SVD norm
            relative_error = sub_vs_svd_error / svd_norm
            
            results[rank]['subspace'][num_iter] = {
                'vs_svd_error': sub_vs_svd_error,
                'relative_to_svd_norm': relative_error
            }
    
    # Print numerical results
    print("\nNumerical Results:")
    print("=================")
    
    for rank in RANKS:
        print(f"\nRank {rank}:")
        print(f"  SVD Approx Norm: {results[rank]['svd_norm']:.3f}")
        
        print(f"  Subspace Iteration Errors (compared to SVD approx):")
        for iter in ITERS:
            sub_vs_svd_error = results[rank]['subspace'][iter]['vs_svd_error']
            rel_error = results[rank]['subspace'][iter]['relative_to_svd_norm']
            print(f"    Iter={iter}: Error vs SVD={sub_vs_svd_error:.3f}, Relative to SVD Norm={rel_error:.3f}")

if __name__ == "__main__":
    main()
