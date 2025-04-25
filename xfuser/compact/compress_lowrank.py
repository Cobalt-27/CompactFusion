import torch
from xfuser.prof import Profiler
import torch.nn.functional as F # For normalization

def svd(input_tensor: torch.Tensor, rank: int):
    U, S, Vh = torch.linalg.svd(input_tensor.float(), full_matrices=False)
    # Create a diagonal matrix from S since S is 1d
    S_diag = torch.diag(S[0:rank]).to(input_tensor.device)
    # Truncate to rank components
    U = U[:, :rank]
    Vh = Vh[:rank, :]
    return (U @ S_diag).to(input_tensor.dtype), Vh.to(input_tensor.dtype)

@Profiler.prof_func("compact.subspace_iter")
@torch.compile
def subspace_iter(
    A: torch.Tensor, rank: int, num_iters: int = 10, init_q: torch.Tensor | None = None
):
    """
    Computes a low-rank approximation of the matrix A using subspace iteration.

    Given A (of shape m x n), the function returns two matrices U (m x rank) and
    V (rank x n) such that A ≈ U V.

    Args:
        A (torch.Tensor): The input matrix of shape (m, n).
        rank (int): The target rank for the low-rank approximation.
        num_iters (int): Number of subspace iteration steps.

    Returns:
        U (torch.Tensor): An orthonormal matrix of shape (m, rank).
        V (torch.Tensor): A matrix of shape (rank, n) such that A ≈ U V.
    """
    m, n = A.shape
    device = A.device
    dtype = A.dtype
    A = A.float()
    # Step 1: Initialize a random matrix Q of shape (n, rank) and orthonormalize it.
    # with Profiler.scope("subspace_iter.init_q"): # avg=0.08ms
    if init_q is None:
        Q = torch.randn(n, rank, device=device, dtype=torch.float)
        Q, _ = torch.linalg.qr(Q)  # Q is (n, rank) 
    else:
        Q = init_q.float()

    # Step 2: Perform subspace iteration on the right subspace.
    for i in range(num_iters):
        # with Profiler.scope("subspace_iter.A_t_A_Q"):
        Z = A.t() @ (A @ Q)  # Shape: (n, rank) avg=0.10ms
        # with Profiler.scope("subspace_iter.qr"):
        Q, _ = torch.linalg.qr(Z) # 0.08ms

    # Step 3: Recover the left singular subspace.
    # Compute U = A Q, then orthonormalize U.
    U_temp = A @ Q  # Shape: (m, rank)
    U, _ = torch.linalg.qr(U_temp)

    # Step 4: Form the low-rank factor V.
    # We set V = U^T A so that A ≈ U V.
    V = U.t() @ A  # Shape: (rank, n)

    return U.to(dtype), V.to(dtype), Q.to(dtype)