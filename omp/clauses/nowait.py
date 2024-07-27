from omp.core.openmp import OpenMP, Clause


@OpenMP.clause('nowait', ('for', 'single'))
class NoWaitClause(Clause):

    name = 'nowait'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        directive.nowait = True
