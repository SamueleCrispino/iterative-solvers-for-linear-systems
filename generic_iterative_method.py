
import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from utils import *

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