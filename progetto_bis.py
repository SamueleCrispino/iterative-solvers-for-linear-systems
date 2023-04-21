import numpy as np

from scipy.io import mmread
from numpy import linalg



def compute_inverted_p(a, n):
    if np.linalg.det(a) == 0:
        # this check also allow as a safe division 1/a.diagonal()
        raise Exception(f"A matrix has a null determinant: can't compute p_1 matrix")

    # init P
    p_1 = np.zeros((n, n))

    np.fill_diagonal(p_1, 1/a.diagonal())
    return p_1

def compute_residue(a, x, b):
    return a.dot(x) - b

def compare_scaled_residue(r, b, tol):
    return linalg.norm(r) / linalg.norm(b) < tol


# generic iterative method:
def generic_iterative_method(a, b, tol, stop='scaled_residue'):
    # TODO: pay attention to /0 operations

    # get A matrix dimensions
    n = a.shape[0]

    # init counter, stop_check, null vector
    k = 0
    stop_check = False
    x = np.zeros(n)


    
    # do i actually need this if condition ??
    if stop == "scaled_residue":
        pass
    
    else:
        # increment over two successive iterations
        pass

    while k < 100 and not stop_check:
        # computing P^-1
        inverted_p = compute_inverted_p(a, n)
        print("inverted_p = ", inverted_p)

        # computing residue
        r = compute_residue(a, x, b)
        print("r = ", r)

        # compunting new x
        x = x - inverted_p.dot(r)
        print("x = ", x)

        # increasing iterations counter
        k = k + 1
        print(k)

        # computing stop check
        stop_check = compare_scaled_residue(r, b, tol)

    print(f"x = {x}")
    print(f"k = {k}")

    return x


a = np.array([5, 2, 3, 4]).reshape(2, 2)
print(a)

b = np.array([16, 18])

generic_iterative_method(a, b, 0.1, stop='scaled_residue')