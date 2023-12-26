"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import random
from collections.abc import Iterable



class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def derive(function, x_val):
            # h = 0.0000000001
            h = 1e-5
            derive = (function(x_val + h) - function(x_val)) / h  # by limit definition
            return derive

        def newton(function, initial, final_bound, interation_bound, eps=maxerr):  # eps = maxerr
            x = (final_bound - initial) / 2  # try x
            if abs(function(initial)) < eps:  # if left index is a root
                return initial

            elif abs(function(final_bound)) < eps:  # if right index is a root
                return final_bound

            for i in range(interation_bound):
                if x < initial or x > final_bound:  # out of segment
                    x = random.uniform(initial, final_bound)  # new x in seg[initial,final]
                if abs(function(x)) < eps:
                    return x  # root
                derive_val = derive(function, x)
                if derive_val == 0:  # derive is 0 - can't divide by 0
                    return bisection(function, initial, final_bound, 20, maxerr)
                x = x - (function(x) / derive_val)
            return  # if no roots have been found in this bound of iterations

        def bisection(function, a, b, interation_bound, eps=maxerr):
            for i in range(interation_bound):
                c = (a + b) / 2  # middle point
                if function(c) == 0 or abs(function(c)) < eps:
                    return c  # root
                elif function(c) * function(a) > 0:
                    a = c
                else:
                    b = c

        all_roots = []
        segments = np.linspace(a, b, 200)  # divide to segments
        function = lambda x: f1(x) - f2(x)
        for i in range(len(segments) - 1):
            start = segments[i]
            end = segments[i + 1]
            if function(start) * function(end) <= 0:  # root exists in this segment
                root = newton(function, segments[i], segments[i + 1], 20, maxerr)
                if root != None:
                    all_roots.append(root)
        return all_roots

# #########################################################################
import unittest
# from sampleFunctions import *

from tqdm import tqdm

#
# class TestAssignment2(unittest.TestCase):
#
#     def test_sqr(self):
#
#         ass2 = Assignment2()
#
#         f1 = np.poly1d([-1, 0, 1])
#         f2 = np.poly1d([1, 0, -1])
#
#         X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#
#         for x in X:
#             print(x)
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
# #
#     def test_poly(self):
#
#         ass2 = Assignment2()
#
#         f1, f2 = randomIntersectingPolynomials(10)
#
#         X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#
#         for x in X:
#             self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
#
# if __name__ == "__main__":
#     unittest.main()
