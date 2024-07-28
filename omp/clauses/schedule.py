from omp.core.openmp import OpenMP, Clause
from omp.core.primitives import Sched

import omp

@OpenMP.clause('schedule', ('for',))
class ScheduleClause(Clause):

    name = 'schedule'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        split_args = args.split(',')

        directive.schedule = (Sched[split_args[0]], int(split_args[-1]) if len(split_args) > 1 else omp.get_schedule()[1])
