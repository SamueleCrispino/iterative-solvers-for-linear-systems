import time
import sys
import numpy as np
from tqdm import tqdm
from scipy.io import mmread
from scipy.sparse import csr_matrix, tril, coo_matrix
from numpy import linalg
from decimal import Decimal

from utils import *
from generic_iterative_method import *


def main(matrix_list, tol_list):
    # execution data summary init
    execution_data = {} 
       
    for matrix in matrix_list:
        file_name = matrix
        a = mmread(file_name)

        for tol in tol_list:
            tol = Decimal(tol)
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
    # TODO: define input handler

    # TODO: Move other subroutines in some subfolder

    # Assuming two lists are passed as command-line arguments
    if len(sys.argv) != 3:
        print('Usage: python script.py matrix_list tol_list')
        sys.exit(1)

    # Extract the lists from command-line arguments
    matrix_list = sys.argv[1].split(',')
    tol_list = sys.argv[2].split(',')

    main(matrix_list, tol_list)











