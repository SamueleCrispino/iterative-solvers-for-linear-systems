import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from implementations.object_oriented_approach.iterative_method import Iterative_method

class Gradient(Iterative_method):

    def __init__(self, tol, a, b, real_x):
        super().__init__(tol, a, b, real_x)
        self.method_name = "GRADIENT_METHOD"
        self.y = None
        self.alfa = None

    def compute_y(self):
        # y = A*d
        self.y = self.a.dot(self.r)

    def compute_gradient_alfa(self):
        self.compute_y()
        self.alfa = self.r.dot(self.r) / (self.r.dot(self.y))

    def compute_next_x(self):   
        self.compute_gradient_alfa() 
        self.x = self.x - self.alfa*self.r
    
        
        
    
