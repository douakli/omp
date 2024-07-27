from omp.core.openmp import OpenMP, Clause


@OpenMP.clause('private', ('critical', 'for', 'parallel', 'single'))
class PrivateClause(Clause):

    name = 'private'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        self.directive.privates.update((varname.strip() for varname in args.split(',')))
