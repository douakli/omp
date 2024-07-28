# Native OpenMP for Python
This library is a proof of concept of a native OpenMP implementation in python.

The `barrier`, `critical`, `for`, `parallel`, `parallel for` and `single` directives are supported,
as well as the `reduction`, `private`, `schedule` and `nowait` clauses.

Here is an example program that uses the library.

```python
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
        with OpenMP("for reduction(+:acc) reduction(*:acc2)"):
            for i in range(1, N):
                acc += i
                acc2 *= i
    print("Actual result:", acc, acc2)
    print("Expected result:", sum(range(1, N)), reduce((lambda a, b: a*b), range(1, N), 1))


if __name__ == '__main__':
    main()
```

<!-- See `examples` for example usages of the library. ->>
