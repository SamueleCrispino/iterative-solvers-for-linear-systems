import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg

# CONSTANTS:
STATIONARY_METHODS = ["jacobi", "Gauß-Seidel"]
NON_STATIONARY_METHODS = ["gradient", "conjugate_gradient"]
METHODS = STATIONARY_METHODS + NON_STATIONARY_METHODS

# METHODS:
def build_sparse_matrix():
    row  = np.array([0, 3, 1, 0, 2])
    col  = np.array([0, 3, 1, 2, 2])
    data = np.array([4, 5, 7, 9, 3])
    a = coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    return a

def compute_rel_error(x, real_x):
    return linalg.norm(x - real_x) / linalg.norm(real_x)

def create_mock(a):    
    real_x = np.ones(a.shape[0])
    b = a.dot(real_x)    

    return real_x, b

def init_values(a, b, n):
    k = 0
    stop_check = False
    x = np.zeros(n)
    max_iter = 20000 if n <= 20000 else n 
    B_NORM = linalg.norm(b)

    return k, stop_check, x, max_iter, B_NORM

def compute_p(a, n, method):

    if method == 'jacobi':
        p_1 = coo_matrix((n, n))
        a_diag = a.diagonal()
        #np.fill_diagonal(p_1, a_diag)
        p_1.setdiag(a_diag)

    elif method == 'Gauß-Seidel':
        p_1 = tril(a) 

    elif method in NON_STATIONARY_METHODS:
        return None

    else:
        raise Exception(f"No options found for method: {method}")

    # check if is non-singular
    if linalg.det(p_1.toarray()) == 0:
        raise Exception(f"Null determinant for a matrix: {p_1}")

    return p_1.tocsr()

def compute_gradient_alfa(r, d_next, z, k, method):
    if k == 0 or method != "conjugate_gradient":
        return r.dot(r) / z
    else:
        return d_next.dot(r) / z


def compute_residue(a, x, b):
    return a.dot(x) - b

def compare_scaled_residue(r, B_NORM, tol):
    return linalg.norm(r) / B_NORM < tol


def input_validation(a, b):
    n = a.shape[0]
    n1 = a.shape[1]
    
    # check a simmetry:
    if not n == n1:
        raise Exception(f"Non-quadratic matrix, dimensions found {n}, {n1}") 

    b_dim = b.shape[0]
    
    # comparing matrix and vector dimensions
    if not b_dim == n1:
        raise Exception(f"Non-comparable dimensions matrix a: {n},{n} and vector b: {b_dim}")

    if not np.all(linalg.eigvals(a) > 0):
        raise Exception(f"a matrix is not positive definite")

    if linalg.det(a) == 0:
        raise Exception(f"Zero determinant for a matrix: {a}, it's singular, not invertible")

    # TODO: add check for diagonal dominance


def forward_substitution(l, r, n):
    y = np.zeros(n)
    pivot = l[0, 0]
    if pivot == 0:
        raise Exception(f"input matrix l: {l} has zero values on diagonal")
    y[0] = r[0] / pivot

    for i in range(1, n):
        pivot = l[i, i]

        if pivot == 0:
            raise Exception(f"input matrix l: {l}  has zero values on diagonal")

        y[i] = (r[i] - l[i].dot(y))/pivot

    return y

def compute_y(a, r, d_next, k, method):
    # y = A*d --> d di questa iterazione = d_next dell'iterazione precedente
    if k == 0 or method != "conjugate_gradient":
        y = a.dot(r)
    else:
        y = a.dot(d_next)
    return y

def compute_next_x(x, alfa, r, d_next, k, method):
    # i've computed r as Ax - b so there i need to substract
    if k == 0 or method != "conjugate_gradient":
        return x - alfa*r
    else:
        return x - alfa*d_next
    
def print_summary(exec_data, method):
    print("*"*50)
    print(f"Summary for {method} method:")
    for k, v in exec_data[method].items():
        if k != "iterations":
            print(f"{k} = {v}")
    print("*"*50)

    
def compute_summary(method, exec_data, k, n, tol, max_iter, start_time, end_time, x, real_x):
    elapsed_time = end_time - start_time
    
    err_rel = compute_rel_error(x, real_x)
    
    exec_data[method]["matrix_dimension"] = n
    exec_data[method]["tol"] = tol
    exec_data[method]["max_iter"] = max_iter
    exec_data[method]["err_rel"] = err_rel
    exec_data[method]["iterations_number"] = k
    exec_data[method]["elapsed_time"] = elapsed_time
    exec_data[method]["iteration_time_avg"] = elapsed_time / k
    
    print_summary(exec_data, method)

    return exec_data

def print_class_summary(class_instance, PARAMS_TO_PRINT):
    for k, v in vars(class_instance).items():
        if k in PARAMS_TO_PRINT:
            print(f"{k} = {v} ")  


