"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""


from sklearn.cluster import KMeans
import numpy as np
import time
import random
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, a):
        # super(MyShape, self).__init__()
        self.area0 = a

    def area(self):
        return self.area0


class Polygon(AbstractShape):
        def __init__(self, knots, noise):
            self._knots = knots
            self._noise = noise
            self._n = len(knots)

        def sample(self):
            i = np.random.randint(self._n)
            t = np.random.random()

            x1, y1 = self._knots[i - 1]
            x2, y2 = self._knots[i]

            x = np.random.random() * (x2 - x1) + x1
            x += np.random.randn() * self._noise

            y = np.random.random() * (y2 - y1) + y1
            y += np.random.randn() * self._noise
            return x, y

        def contour(self, n: int):
            ppf = n // self._n
            rem = n % self._n
            points = []
            for i in range(self._n):
                ts = np.linspace(0, 1, num=(ppf + 2 if i < rem else ppf + 1))

                x1, y1 = self._knots[i - 1]
                x2, y2 = self._knots[i]

                for t in ts[0:-1]:
                    x = t * (x2 - x1) + x1
                    y = t * (y2 - y1) + y1
                    xy = np.array((x, y))
                    points.append(xy)
            points = np.stack(points, axis=0)
            return points

        def area(self):
            a = 0
            for i in range(self._n):
                x1, y1 = self._knots[i - 1]
                x2, y2 = self._knots[i]
                a += 0.5 * (x2 - x1) * (y1 + y2)
            return a


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        def calc_area(points):
            p_num = len(points)
            area = 0
            for i in range(p_num):
                area += (points[i][0] - points[i-1][0]) * (points[i][1] + points[i-1][1])
            area = area/2
            return area

        curr_area = 0
        prev = 0
        err = maxerr + 1
        p_num = 10
        while err > maxerr:
            prev = curr_area
            points = contour(p_num)
            curr_area = calc_area(points)
            p_num = 2 * p_num
            err = abs(prev - curr_area) / curr_area

        return np.float32(curr_area)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        def sort_P(points):
            x = np.array(points)
            center = np.mean(x, axis=0)
            centered = x - center
            angles = np.angle(centered[:, 0] + 1j * centered[:, 1])
            sorted_indices = np.argsort(angles)
            sorted_points = [centered[i] + center for i in sorted_indices]
            return np.array(sorted_points)

        start_time = time.time()
        points = []
        i = 0
        while time.time() - start_time < 0.9*maxtime:
            if i <= 120000:
                point = sample()
                points.append(point)
                i+=1
            else:
                break
        kmeans = KMeans(n_clusters=30, random_state=0).fit(points)
        shape_contour_points = kmeans.cluster_centers_
        sorted_contour = sort_P(shape_contour_points)
        polygon = Polygon(sorted_contour, 0)
        arg = polygon.area()
        res = MyShape(arg)
        return res

##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
