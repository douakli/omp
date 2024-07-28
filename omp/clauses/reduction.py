from omp.core.openmp import OpenMP, Clause


@OpenMP.clause('reduction', ('for'))
class ReductionClause(Clause):

    name = 'reduction'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        operator, varname = args.split(':')

        self.directive.privates.add(varname.strip())
        self.directive.reduction.update({varname: operator})


operators = {
    '+': lambda a, b: a + b,
    '*': lambda a, b: a * b,
    '&': lambda a, b: a & b,
    '|': lambda a, b: a | b,
    '^': lambda a, b: a ^ b,
    '&&': lambda a, b: a and b,
    '||': lambda a, b: a or b,
    None: lambda a, b: a
}
