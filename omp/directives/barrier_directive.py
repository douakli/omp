from omp.core.openmp import Directive, OpenMP
from omp.core.threading import barrier

import ast


@OpenMP.directive('barrier')
class BarrierDirective(Directive):

    """
    OpenMP parallel construct implementation.
    """

    def parse(self, node: ast.With) -> ast.With:
        return node

    def run(self):
        barrier()
