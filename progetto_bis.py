import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg

# CONSTANTS
STATIONARY_METHODS = ["jacobi", "Gauß-Seidel"]
NON_STATIONARY_METHODS = ["gradient", "conjugate_gradient"]
METHODS = STATIONARY_METHODS + NON_STATIONARY_METHODS


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

def compute_gradient_alfa(a, r, y, d_next):
    if d_next == None:
        return r.dot(r) / (r.dot(y))
    else:
        return d_next.dot(r) / d_next.dot(y)


def compute_residue(a, x, b):
    return a.dot(x) - b

def compare_scaled_residue(r, B_NORM, tol):
    return linalg.norm(r) / B_NORM < tol


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

def compute_y(a, r, d_next):
    # y = A*d
    if d_next == None:
        y = a.dot(r)
    else:
        y = a.dot(d_next)

    return y

def compute_next_x(x, alfa, r, d_next):
    if d_next == None:
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



# generic iterative method:
def generic_iterative_method(a, b, real_x, method, exec_data, tol, validation=False):
    # TODO: pay attention to /0 operations  
    exec_data[method] = {}
    exec_data[method]["iterations"] = {}

    # Start the timer
    start_time = time.time()  

    print(f"Starting iterative {method} method")

    #TODO: this can be done outside the iterative method
    if validation:
        input_validation(a, b)
    
    #TODO: this can be done outside the iterative method
    # get A matrix dimensions
    n = a.shape[0]

    #TODO: this can be done outside the iterative method
    # init counter, stop_check, null vector, max_iter, real_X, b vector
    k, stop_check, x, max_iter, B_NORM = init_values(a, b, n)

    # case b = null and assuming that determinant of a matrix is different from zero
    if not np.any(b):
        print("A null b vector is passed")
        return x

    p = compute_p(a, n, method)

    with tqdm(total = max_iter) as pbar:
        while k <= max_iter and not stop_check:
            r_next = None
            d_next = None

            # computing residue
            r = compute_residue(a, x, b) if r_next == None else r_next
            #print("r = ", r)

            # compunting new x
            if method in STATIONARY_METHODS:
                x = x - forward_substitution(p, r, n)
            if method in NON_STATIONARY_METHODS:
                y = compute_y(a, r, d_next)
                alfa = compute_gradient_alfa(a, r, y, d_next)
                
                # i've computed r as Ax - b so there i need to substract
                x = compute_next_x(x, alfa, r, d_next)
                
                if method == "conjugate_gradient":
                    r_next = compute_residue(a, x, b)
                    w = a.dot(r_next)
                    beta = r.dot(w) / r.dot(y)
                    d_next = r_next - beta*r
            

            # increasing iterations counter
            k = k + 1
            exec_data[method]["iterations"][k] = r

            # computing stop check
            stop_check = compare_scaled_residue(r, B_NORM, tol)
            
            pbar.update(1)

    # End the timer and elapsed time in seconds
    end_time = time.time()
    
    if k > max_iter:
        print(f"Exceeded max number of iterations: {max_iter}")

    exec_data = compute_summary(method, exec_data, k, n, tol, max_iter,
                                start_time, end_time, x, real_x)


    return x, exec_data


def main(matrix_list, tol_list):
    # execution data summary init
    execution_data = {} 
       
    for matrix in matrix_list:
        file_name = matrix
        a = mmread(file_name)

        for tol in tol_list:
            # real solution
            real_x, b = create_mock(a)           
            
            matrix_name = file_name.split("/")[-1]

            if matrix_name not in execution_data:
                execution_data[matrix_name] = {}
                print(f"matrix name: {matrix_name}")
            
            if tol not in execution_data[matrix_name]:
                execution_data[matrix_name][tol] = {}
                print(f"tol: {tol}")
    
            print("*"*50)
            print("*"*50)

            for method in METHODS:
                x, execution_data[matrix_name][tol] = generic_iterative_method(a, b, real_x, method, 
                                                    execution_data[matrix_name][tol], tol, 
                                                    validation=False)
        
    return execution_data


if __name__ == "__main__":
    # Assuming two lists are passed as command-line arguments
    if len(sys.argv) != 3:
        print('Usage: python script.py matrix_list tol_list')
        sys.exit(1)

    # Extract the lists from command-line arguments
    matrix_list = sys.argv[1].split(',')
    tol_list = sys.argv[2].split(',')

    main(matrix_list, tol_list)











