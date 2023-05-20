import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from implementations.object_oriented_approach.iterative_method import Iterative_method

class Jacobi(Iterative_method):
    def __init__(self, tol, a, b, real_x):
        super().__init__(tol, a, b, real_x)
        self.method_name = "JACOBI_METHOD"
        self.p = None

    def forward_substitution(self):
        excep_message = f"input matrix p: has zero values on diagonal"
        
        y = np.zeros(self.n)
        pivot = self.p[0, 0]
        if pivot == 0:
            raise Exception(excep_message)
        y[0] = self.r[0] / pivot

        for i in range(1, self.n):
            pivot = self.p[i, i]
            if pivot == 0:
                raise Exception(excep_message)

            y[i] = (self.r[i] - self.p[i].dot(y))/pivot
        
        return y

    def before_iterations(self):
        p_1 = coo_matrix((self.n, self.n))
        a_diag = self.a.diagonal()
        p_1.setdiag(a_diag)

        # check if is non-singular
        if linalg.det(p_1.toarray()) == 0:
            raise Exception(f"Null determinant for a matrix: {p_1}")
        self.p = p_1.tocsr()

    def compute_next_x(self):
        self.x = self.x - self.forward_substitution()
