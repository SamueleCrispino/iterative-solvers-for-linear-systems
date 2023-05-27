import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from implementations.utils.utils import *

from implementations.object_oriented_approach.stationary_methods.jacobi import Jacobi
from implementations.object_oriented_approach.stationary_methods.gauß_seidel import Gauß_Seidel
from implementations.object_oriented_approach.non_stationary_methods.gradient import Gradient
from implementations.object_oriented_approach.non_stationary_methods.conjugate_gradient import Conjugate_gradient


PARAMS_TO_PRINT = ["k", "method_name", "elapsed_time", "iteration_time_avg", "err_rel"]


def main(file_name, tol):
    a = mmread(file_name)
    real_x, b = create_mock(a)    
    
    jacobi = Jacobi(tol, a, b, real_x)
    solution = jacobi.run()
    print_class_summary(jacobi, PARAMS_TO_PRINT)
    
    gs = Gauß_Seidel(tol, a, b, real_x)
    solution = gs.run()
    print_class_summary(gs, PARAMS_TO_PRINT)
    
    gradient = Gradient(tol, a, b, real_x)
    solution = gradient.run()
    print_class_summary(gradient, PARAMS_TO_PRINT)

    conj_grad = Conjugate_gradient(tol, a, b, real_x)
    solution = conj_grad.run()
    print_class_summary(conj_grad, PARAMS_TO_PRINT)

    return jacobi, gs, gradient, conj_grad


if __name__ == "__main__":
    try:
        file_name = sys.argv[1]
        tol = Decimal(sys.argv[2])
        
        main(file_name, tol)
    except Exception as e:
        print(e)
        
