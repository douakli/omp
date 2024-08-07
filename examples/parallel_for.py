#!/usr/bin/env python3
import omp
from omp import OpenMP

N = 400000000


@omp.enable
def main():
    acc = 0
    with OpenMP("parallel"):
        with OpenMP("for reduction(+:acc) schedule(dynamic, 10000)"):
            for i in range(1, N):
                acc += i
    print("Actual result:  ", acc)
    print("Expected result:", N*(N-1)//2)


if __name__ == '__main__':
    main()
