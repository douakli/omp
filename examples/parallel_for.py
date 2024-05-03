#!/usr/bin/env python3
import omp
from omp import OpenMP

import threading

N = 40


@omp.enable
def main():

    # Declare a list of N elements.
    shared_list = [None] * N

    # Start a team of threads.
    with OpenMP("parallel"):

        # Distribute the following for loop across the threads of the team.
        with OpenMP("for"):
            # We only do as many iterations as there are threads in the team, showcasing that there are indeed different threads running the loop.
            for i in range(threading.current_thread().team.size):
                # Show the current thread rank.
                print(f"Hello from thread {threading.current_thread().rank}")

        # Distribute the following for loop across the threads of the team.
        with OpenMP("for"):
            for i in range(N):
                shared_list[i] = i

    # Show the resulting list
    print(shared_list)


if __name__ == '__main__':
    main()
    pass
