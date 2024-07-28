from omp.core.openmp import OpenMP, Clause
from omp.core.primitives import Sched


@OpenMP.clause('schedule', ('for',))
class ScheduleClause(Clause):

    name = 'schedule'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        directive.schedule = Sched[args]
