import sys
from typing import Self
import math


"""
A custom vector class implementation for educational purposes.
"""


class Vec:
    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = ()
        else:
            self.elements = tuple(src)

    def scalar_mult(self, alpha):
        el = list(self.elements)
        for i in range(len(el)):
            el[i] = alpha * el[i]

        return el

    # Implement scalar multiplication without creating a temp list
    def scalar_mult1(self, alpha):
        return Vec(alpha * element for element in self.elements)

    def mean(self):
        total = sum(self.elements)
        count = len(self.elements)
        mean = total / count
        return mean

    def demean(self):
        mean_value = self.mean()
        return Vec(element - mean_value for element in self.elements)

    def std(self):
        demeaned_vector = self.demean()

        squared_deviations = (
            element ** 2 for element in demeaned_vector.elements
        )

        average_squared_deviation = (
            sum(squared_deviations) / len(self.elements)
        )

        return math.sqrt(average_squared_deviation)

    def __repr__(self):
        return "myvector:" + repr(self.elements)


"""
(1) Understand the basic design of the vector abstraction. Review the implementation.
(2) Document each function.
(3) Implement all unimplemented methods.
(4) Create appropriate tests for this implementation, increasing the confidence about its correctness.

(5) Test this implementation by importing the class in a separate python script.

(6) Measure the performance of each of these functions on vectors of varying lengths.
    Try 2k to 64k dimension vectors and time the results.
    How would you do the measurements?

(7) Measure the performance on your machine. Check it on colab.

(8) Use numpy and compare the performance.
"""


if sys.version_info < (3, 8):
    sys.exit("Error: This script requires Python 3.8 or higher.")


if __name__ == "__main__":
    v1 = Vec([0, 1, 1.03])

    print(v1)
    print(v1.scalar_mult(2))
    print(v1.scalar_mult1(4))

    print("Mean:", v1.mean())
    print("Demeaned:", v1.demean())
    print("Standard deviation:", v1.std())