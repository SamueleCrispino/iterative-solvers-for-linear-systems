import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal


class Iterative_method:
    def __init__(self, tol, a, b, real_x):
        # class variables for reporting
        self.method_name = "GENERIC_ITERATIVE_METHOD"
        self.start_time = None
        self.elapsed_time = None
        self.err_rel = None
        self.iteration_time_avg = None
        self.convergence = {}
        self.abs_convergence = {}
        
        # class variables start parameters
        self.a = a
        self.b = b
        self.tol = tol
        self.real_x = real_x
        self.n = a.shape[0]        
        self.stop_check = False        
        self.max_iter = 20000 if self.n <= 20000 else self.n
        self.B_NORM = linalg.norm(b)

        # class variables dynamic parameters
        self.k = 0
        self.x = np.zeros(self.n)
        self.r = None
        self.scaled_residue = None


    def compute_scaled_residue(self):
        self.scaled_residue = linalg.norm(self.r) / self.B_NORM

    def compute_avg_time_per_iteration(self):
        self.iteration_time_avg = self.elapsed_time / self.k

    def compute_elapsed_time(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time 

    def compute_rel_error(self):
        self.err_rel = linalg.norm(self.x - self.real_x) / linalg.norm(self.real_x)

    def compute_residue(self):
        self.r = self.a.dot(self.x) - self.b
    
    def compare_scaled_residue(self):
        self.compute_scaled_residue()
        self.stop_check = self.scaled_residue < self.tol

    def update_reporting_parameters(self):
        self.compare_scaled_residue()
        self.k+=1 
        self.convergence[self.k] = self.scaled_residue
        self.abs_convergence[self.k] = abs(self.scaled_residue)
        
    def compute_reporting_summary(self):
        self.compute_rel_error()
        self.compute_elapsed_time()
        self.compute_avg_time_per_iteration()

    def after_iterations(self):
        self.compute_reporting_summary()
        
        if self.k > self.max_iter:
            print(f"Exceeded max number of iterations: {self.max_iter}")
            return None, self.exec_data
        else:
            return self.x

    def before_iterations(self):
        pass

    def compute_next_x(self):
        pass
    
    def iterate(self):

        self.start_time = time.time()

        with tqdm(total = self.max_iter) as pbar:
            while self.k <= self.max_iter and not self.stop_check:

                self.compute_residue()
                self.compute_next_x()                
                self.update_reporting_parameters()

                pbar.update(1)

    def run(self):
        self.before_iterations()
        self.iterate()
        return self.after_iterations()






