"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import math
import numpy as np
import time
import random
import assignment2
from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        if n == 2:
            result = ((f(a) + f(b)) / 2) * (b - a)
            return np.float32(result)
        f0 = 0
        f1 = 0
        edges = f(a) + f(b)

        if n % 2 == 0:
            n = n - 1
        h = (b - a) / (n-1)

        for i in range(2, n - 1, 2):  # even points
            x = a + i * h
            f0 += f(x)
        for i in range(1, n, 2):  # odd points
            x = a + i * h
            f1 += f(x)

        total_calc = (h / 3) * (2 * f0 + 4 * f1 + edges)
        result = np.float32(total_calc)
        return result

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """
        ass2 = assignment2.Assignment2()
        intersection_points = list(ass2.intersections(f1, f2, 1, 100))
        all_area = 0
        if len(intersection_points) < 2: # no area between the 2 functions
            return np.nan
        all_inter = len(intersection_points)
        for i in range(all_inter-1):
            all_area += self.integrate(lambda x: abs(f1(x) - f2(x)), intersection_points[i], intersection_points[i+1], 100)
        result = np.float32(all_area)
        return result


##########################################################################
# ass3 = Assignment3()
# print(ass3.areabetween(lambda x:x**2-4*x, lambda y:y-5))
# # #
# import unittest
# from sampleFunctions import *
# import math
# from tqdm import tqdm
#
#
# class TestAssignment3(unittest.TestCase):
#
#     def test_integrate_float32(self):
#         ass3 = Assignment3()
#         # f1 = np.poly1d([-1, 0, 1])
#         f1= lambda x: np.sin(x)
#         r = ass3.integrate(f1, -10, 60, 58)
#         print("sinx integral area is " + str(r))
#         self.assertEqual(r.dtype, np.float32)
#
#     def test1_integrate_float32(self):
#         ass3 = Assignment3()
#         f1 = np.poly1d([1, 0, 0, 4])
#         r = ass3.integrate(f1, 0, 2, 69)
#         print("the answer is " + str(r))
#         self.assertEqual(r.dtype, np.float32)
#
#     def test_integrate_hard_case(self):
#         ass3 = Assignment3()
#         f1 = strong_oscilations()
#         r = ass3.integrate(f1, 0.09, 10, 200000)
#         true_result = -7.78662 * 10 ** 33
#         print(r)
#         print(true_result)
#         self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))
#     def test_area_between(self):
#         ass3 = Assignment3()
#         f1 = lambda x: np.sin(x)
#         f2 = lambda x: np.cos(x)
#         r = ass3.areabetween(f1, f2)
#         print("tha ans is" + str(r))



# if __name__ == "__main__":
#     unittest.main()
