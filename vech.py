import torch

def vech(A: torch.Tensor) -> torch.Tensor: # needed for RNN -> BEKK parameter matrices (C*C^T)
    """This function takes a square matrix A and returns a vector containing the elements of the lower triangular 
       part of A (including the diagonal), stacked column-wise.

    Args:
        A (torch.Tensor): Square matrix of shape (d, d)

    Returns:
        torch.Tensor: Vector of shape (d(d+1)/2,) containing the lower-triangular elements of A
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"vech expects square matrix input on the last 2 dims, got shape {tuple(A.shape)}")

    d = A.shape[-1]
    lower_tri = []
    for j in range(d):
        lower_tri.append(A[..., j:, j])

    # Keep all leading batch dims and flatten only lower-triangular entries.
    return torch.cat(lower_tri, dim=-1)


# nach letztem h_t mit Bxh und dem linear layer zu Bx(d(d+1)/2) -> dann mit unvech() zu Bxdxd
def unvech(v: torch.Tensor, d: int) -> torch.Tensor: # needed for RNN -> covariance matrix
    """This is a function that reconstructs a square matrix from its lower-triangular elements provided in a vector.
        Basically the inverse of vech().

    Args:
        v (torch.Tensor): Vector of shape (..., d(d+1)/2,) containing the lower-triangular elements of a matrix
        d (int): Dimension of the square matrix to be reconstructed

    Returns:
        torch.Tensor: Square matrix of shape (..., d, d) reconstructed from the lower-triangular elements
    """
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")

    expected = d * (d + 1) // 2
    if v.shape[-1] != expected:
        raise ValueError(
            f"unvech expected last dimension {expected} for d={d}, got {v.shape[-1]} "
            f"(input shape: {tuple(v.shape)})"
        )

    A = v.new_zeros(*v.shape[:-1], d, d)
    k = 0
    for j in range(d):
        n = d - j
        A[..., j:, j] = v[..., k:k+n]
        k += n
    return A