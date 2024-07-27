#!/usr/bin/env python3
import omp
from omp import OpenMP

from functools import reduce

N = 20


@omp.enable
def main():
    acc = 0
    acc2 = 1
    with OpenMP("parallel"):
        with OpenMP("for collapse(+:acc) collapse(*:acc2)"):
            for i in range(1, N):
                acc += i
                acc2 *= i
    print("Actual result:", acc, acc2)
    print("Expected result:", sum(range(1, N)), reduce((lambda a, b: a*b), range(1, N), 1))


if __name__ == '__main__':
    main()
