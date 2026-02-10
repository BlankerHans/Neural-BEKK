import torch

def vech(A: torch.Tensor) -> torch.Tensor: # needed for RNN -> BEKK parameter matrices (C*C^T)
    """This function takes a square matrix A and returns a vector containing the elements of the lower triangular 
       part of A (including the diagonal), stacked column-wise.

    Args:
        A (torch.Tensor): Square matrix of shape (d, d)

    Returns:
        torch.Tensor: Vector of shape (d(d+1)/2,) containing the lower-triangular elements of A
    """
    d = A.shape[-1]
    lower_tri = []
    for j in range(d):
        lower_tri.append(A[j:, j])
    
    return torch.cat(lower_tri)

def unvech(v: torch.tensor, d: int) -> torch.Tensor: # needed for RNN -> covariance matrix
    """This is a function that reconstructs a square matrix from its lower-triangular elements provided in a vector.
        Basically the inverse of vech().

    Args:
        v (torch.tensor): Vector of shape (d(d+1)/2,) containing the lower-triangular elements of a matrix
        d (int): Dimension of the square matrix to be reconstructed

    Returns:
        torch.Tensor: Square matrix of shape (d, d) reconstructed from the lower-triangular elements
    """
    A = torch.zeros(d, d, device=v.device)
    k = 0
    for j in range(d):
        n = d - j # Anzahl Elemente in der j-ten Spalte
        A[j:, j] = v[k:k+n] # k Startindex für j-te Spalte, muss da anfangen, wo die vorherige Spalte aufgehört hat
        k += n
        print(k)
    return A