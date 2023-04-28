import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg

def compute_rel_error(x, real_x):
    return linalg.norm(x - real_x) / linalg.norm(real_x)

def create_mock(a):    
    real_x = np.ones(a.shape[0])
    b = a.dot(real_x)    

    return real_x, b

def init_values(a, n):
    k = 0
    stop_check = False
    x = np.zeros(n)
    max_iter = 2000 if n <= 2000 else n

    return k, stop_check, x, max_iter

def compute_p(a, n, method):

    if method == 'jacobi':
        p_1 = coo_matrix((n, n))
        a_diag = a.diagonal()
        #np.fill_diagonal(p_1, a_diag)
        p_1.setdiag(a_diag)

    elif method == 'GS':
        p_1 = tril(a)  

    else:
        raise Exception(f"No options found for method: {method}")

    # check if is non-singular
    if linalg.det(p_1.toarray()) == 0:
        raise Exception(f"Null determinant for a matrix: {p_1}")

    return p_1.tocsr()



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


def forward_substitution(l, r, n):
    y = np.zeros(n)

    if l[0, 0] == 0:
        raise Exception(f"input matrix l: {l} has zero values on diagonal")
    y[0] = r[0] / l[0, 0]

    for i in range(1, n):
        if l[i, i] == 0:
            raise Exception(f"input matrix l: {l}  has zero values on diagonal")

        y[i] = (r[i] - l[i].dot(y))/l[i, i] 

    return y


# generic iterative method:
def generic_iterative_method(a, b, real_x, method, tol=0.0001, validation=False):
    # TODO: pay attention to /0 operations

    print("Starting iterative method")

    if validation:
        n = input_validation(a, b)
    else:
        # get A matrix dimensions
        n = a.shape[0]


    # init counter, stop_check, null vector, max_iter, real_X, b vector
    k, stop_check, x, max_iter = init_values(a, n)
    
    print(f"tol = {tol}")
    print(f"max_iter = {max_iter}")

    # case b = null and assuming that determinant of a matrix is different from zero
    if not np.any(b):
        print("A null b vector is passed")
        return x

    p = compute_p(a, n, method)

    with tqdm(total = max_iter) as pbar:
        while k <= max_iter and not stop_check:

            # computing residue
            r = compute_residue(a, x, b)
            #print("r = ", r)

            # compunting new x
            x = x - forward_substitution(p, r, n)
            #print("x = ", x)

            # increasing iterations counter
            k = k + 1
            #print(k)

            # computing stop check
            stop_check = compare_scaled_residue(r, b, tol)
            
            pbar.update(1)

    # TODO: saving these for plot
    

    if k > max_iter:
        print(f"Exceeded max number of iterations: {max_iter}")

    print(f"x = {x}")
    print(f"k = {k}")
    err_rel = compute_rel_error(x, real_x)
    print(f"err_rel = {err_rel}")

    return x

def main():
    a = mmread('data/vem1.mtx')

    #a = coo_matrix(np.array([5, 2, 3, 4]).reshape(2, 2))
    
    real_x, b = create_mock(a)

    generic_iterative_method(a, b, real_x, 'GS', validation=False)



if __name__ == "__main__":
    main()

#a = np.array([5, 2, 3, 4]).reshape(2, 2)
#b = np.array([30, 46])








