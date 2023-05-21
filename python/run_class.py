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
    
    solver = Jacobi(tol, a, b, real_x)
    solution = solver.run()
    print_class_summary(solver, PARAMS_TO_PRINT)

    solver = Gauß_Seidel(tol, a, b, real_x)
    solution = solver.run()
    print_class_summary(solver, PARAMS_TO_PRINT)

    solver = Gradient(tol, a, b, real_x)
    solution = solver.run()
    print_class_summary(solver, PARAMS_TO_PRINT)

    solver = Conjugate_gradient(tol, a, b, real_x)
    solution = solver.run()
    print_class_summary(solver, PARAMS_TO_PRINT)


if __name__ == "__main__":
    file_name = "../data/spa1.mtx"
    tol = 10**-4
    main(file_name, tol)
