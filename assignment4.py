"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import numpy as np
import time
import random
from numpy.linalg import inv


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        start_time = time.time()
        y = []
        x_new = []
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > maxtime:
                break
            else:

                try:
                    x_point = np.random.uniform(a, b, 1)[0]
                    y.append(f(x_point))
                    x_new.append(x_point)
                except ValueError as e:
                    if str(e) == "math domain error":
                        continue
                except ZeroDivisionError:
                    continue
        B = np.array(y)
        A = np.vander(x_new, d+1)
        A_transposed = np.transpose(A)
        AT_A = np.dot(A_transposed, A)
        AT_B = np.dot(A_transposed, B)
        A_inv = np.linalg.inv(AT_A)
        m = np.dot(A_inv, AT_B)
        return lambda x: np.float32(np.poly1d(m)(x))


##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm
#
#
# class TestAssignment4(unittest.TestCase):
# #
#     def test_return(self):
#         f = NOISY(0.01)(poly(1,1,1))
#         ass4 = Assignment4()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertLessEqual(T, 5)
# #
#     def test_delay(self):
#         f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
#
#         ass4 = Assignment4()
#         T = time.time()
#         shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         self.assertGreaterEqual(T, 5)
# #
#     def test_err(self):
#         f = poly(1,1,1)
#         nf = NOISY(1)(f)
#         ass4 = Assignment4()
#         T = time.time()
#         ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
#         T = time.time() - T
#         mse=0
#         for x in np.linspace(0,1,1000):
#             self.assertNotEqual(f(x), nf(x))
#             mse+= (f(x)-ff(x))**2
#         mse = mse/1000
#         print(mse)
#
# #
#

#
# if __name__ == "__main__":
#     unittest.main()
