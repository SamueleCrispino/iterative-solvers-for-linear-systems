import numpy as np

from scipy.io import mmread
from numpy import linalg



def compute_inverted_p(a, n):
    # init P
    p_1 = np.zeros((n, n))

    a_diag = a.diagonal()

    if 0 in a_diag:
        raise Exception(f"P matrix has a null determinant, hence it's singular and can't compute p_1 matrix")

    np.fill_diagonal(p_1, 1/a.diagonal())
    
    return p_1

def compute_residue(a, x, b):
    return a.dot(x) - b

def compare_scaled_residue(r, b, tol):
    return linalg.norm(r) / linalg.norm(b) < tol


def input_validation(a, b):
    n = a.shape[0]
    n1 = a.shape[1]
    
    # check a simmetry:
    if not n == n1:
        raise Exception(f"Non-symmetric a matrix, dimensions found {n}, {n1}") 

    b_dim = b.shape[0]
    
    # comparing matrix and vector dimensions
    if not b_dim == n1:
        raise Exception(f"Non-comparable dimensions matrix a: {n},{n} and vector b: {b_dim}")

    if not np.all(linalg.eigvals(a) > 0):
        raise Exception(f"a matrix is not positive definite")

    if linalg.det(a) == 0:
        raise Exception(f"Negative determinant for a matrix: {a}")

    # TODO: add check for diagonal dominance
    

    return n


# generic iterative method:
def generic_iterative_method(a, b, tol=0.0001, stop='scaled_residue', validation=False):
    # TODO: pay attention to /0 operations

    if validation:
        n = input_validation(a, b)
    else:
        # get A matrix dimensions
        n = a.shape[0]

    # init counter, stop_check, null vector, max_iter
    k = 0
    stop_check = False
    x = np.zeros(n)
    max_iter = 2000 if n <= 2000 else n

    # case b = null and assuming that determinant of a matrix is different from zero
    if not np.any(b):
        print("A null b vector is passed")
        return x

    
    # do i actually need this if condition ??
    if stop == "scaled_residue":
        pass
    
    else:
        # increment over two successive iterations
        pass

    # computing P^-1
    inverted_p = compute_inverted_p(a, n)
    print("inverted_p = ", inverted_p)

    while k <= max_iter and not stop_check:

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

    # TODO: saving these for plot
    print(f"x = {x}")
    print(f"k = {k}")
    print(f"tol = {tol}")

    if k > max_iter:
        print(f"Exceeded max number of iterations: {k}")

    return x


a = np.array([5, 2, 3, 4]).reshape(2, 2)
b = np.array([30, 46])

generic_iterative_method(a, b, stop='scaled_residue', validation=True)