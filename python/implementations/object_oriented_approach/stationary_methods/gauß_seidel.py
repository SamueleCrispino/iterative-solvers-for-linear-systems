import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from implementations.object_oriented_approach.stationary_methods.jacobi import Jacobi

class Gauß_Seidel(Jacobi):

    def __init__(self, tol, a, b, real_x):
        super().__init__(tol, a, b, real_x)
        self.method_name = "GAUß_SEIDEL_METHOD"

    def before_iterations(self):
        p_1 = tril(self.a) 

        # check if is non-singular
        if linalg.det(p_1.toarray()) == 0:
            raise Exception(f"Null determinant for a matrix: {p_1}")
        self.p = p_1.tocsr()