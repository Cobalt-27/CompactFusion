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

_rand_vectors = {} 
def subspace_iter(
    A: torch.Tensor, rank: int, num_iters: int = 10, init_q: torch.Tensor | None = None
):
    """
    Computes a low-rank approximation of the matrix A using subspace iteration.
    """
    RAND_LIST_SIZE = 1024
    if rank == 1:
        n = A.shape[1]
        if n not in _rand_vectors:
            rand = torch.randn(RAND_LIST_SIZE,n, 1, device=A.device, dtype=torch.float)
            #normalize each vector
            rand = rand / torch.norm(rand, dim=1, keepdim=True)
            _rand_vectors[n] = rand
        v = _rand_vectors[n][torch.randint(0, RAND_LIST_SIZE, (1,))] # Shape (1, n, 1)
        v = v.squeeze(0) # Shape (n, 1)
        return _power_iteration(A, num_iters, v)
    else:
        return _subspace_iter(A, rank, num_iters, init_q)



@torch.jit.script
def _subspace_iter(
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

# @Profiler.prof_func("compact.power_iteration")
@torch.jit.script
def _power_iteration(
    A: torch.Tensor, num_iters: int, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a rank-1 approximation A ≈ U @ V using power iteration.
    This focuses on finding the factors U and V directly.

    Args:
        A (torch.Tensor): The input matrix of shape (m, n).
        num_iters (int): Number of power iteration steps.

    Returns:
        U (torch.Tensor): The left factor, shape (m, 1). Contains the singular value scale.
        V (torch.Tensor): The right factor (transposed right singular vector), shape (1, n).
    """
    m, n = A.shape
    device = A.device
    input_dtype = A.dtype

    # Initialize a random vector for the right singular vector v
    # v = torch.randn(n, 1, device=device, dtype=torch.float)
    # v = v / torch.norm(v, dim=0) # Normalize v

    A_float = A.float() # Perform iterations in float32 for stability

    # Power iteration loop focused on finding the dominant right singular vector v
    for _ in range(num_iters):
        # Equivalent to A.t() @ A @ v iteration step
        v_unnormalized = A_float.t() @ (A_float @ v) # Shape (n, 1)
        # Normalize v for the next iteration
        v = v_unnormalized / torch.norm(v_unnormalized, dim=0) # Add epsilon for stability

    # After iterations, v approximates the dominant right singular vector.
    # Calculate U = A @ v. This U effectively contains the scale (singular value). U = u * s
    U_float = A_float @ v # Shape (m, 1)

    # V is the conjugate transpose of the normalized right singular vector v. V = v.t()
    V_float = v.t() # Shape (1, n)

    # Return U, V converted back to original dtype
    U = U_float.to(input_dtype)
    V = V_float.to(input_dtype)

    return U, V, v
