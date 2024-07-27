from omp.core.openmp import OpenMP, Clause


@OpenMP.clause('collapse', ('for'))
class CollapseClause(Clause):

    name = 'collapse'

    def __init__(self, directive, args):
        super().__init__(directive, args)

        operator, varname = args.split(':')

        self.directive.privates.add(varname.strip())
        self.directive.collapse.update({varname: operator})


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
