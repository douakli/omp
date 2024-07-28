#!/usr/bin/env python3
import omp
from omp import OpenMP

N = 400000


@omp.enable
def main():
    acc = 0
    with OpenMP("parallel"):
        with OpenMP("for reduction(+:acc) schedule(static)"):
            for i in range(1, N):
                acc += i
    print("Actual result:  ", acc)
    print("Expected result:", sum(range(1, N)))


if __name__ == '__main__':
    main()
