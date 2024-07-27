import omp
from omp.core.openmp import Directive, OpenMP
from omp.core.ast_tools import LinenoStripper

import ast
import random


def generator(it):
    """
    When called within a thread of a team, yields the iterations for the current thread.

    Overall, when all the threads of the team call this generator, all the elements of the iterator are yielded.
    """
    for i, el in enumerate(it):
        if i % omp.get_num_threads() == omp.get_thread_num():
            yield el


@OpenMP.directive('for')
class ForConstruct(Directive):

    """
    OpenMP for construct implementation.
    """
    @property
    def template(self):

        collapse_vars, collapse_operators = list(zip(*self.collapse.items()))
        nonce = random.randint(0, 100000)

        return f"""\
with _omp_internal.core.openmp.OpenMP():
    if False:
        pass # Replaced by shared variables declarations
    def _omp_internal_inner_func{nonce}({','.join(collapse_vars)}):
        pass # Replaced by user code
        {'#' if self.nowait else ''}_omp_internal.core.openmp.OpenMP("barrier")
        return ({','.join(collapse_vars)},)
    def _omp_internal_inner_func_protect{nonce}():
        nonlocal {','.join(collapse_vars)}
        _omp_internal_retval = _omp_internal_inner_func{nonce}({','.join(collapse_vars)})
        with _omp_internal.core.openmp.OpenMP('critical'):
            ({','.join(collapse_vars)},) = [_omp_internal.clauses.collapse.operators[_omp_internal_operator](_omp_internal_var, _omp_internal_val) for _omp_internal_operator, _omp_internal_var, _omp_internal_val in zip({collapse_operators}, ({','.join(collapse_vars)},), _omp_internal_retval)]
    _omp_internal_inner_func_protect{nonce}()
    """

    def parse(self, node: ast.With):
        for_node: ast.For = node.body[0]

        # Wrap the loop iterator in our thread-distributing generator.
        for_node.iter = ast.Call(LinenoStripper().visit(ast.parse('_omp_internal.directives.for_construct.generator')).body[0].value, args=[for_node.iter], keywords=[])

        # We need to protect the target.
        # ALERT: We need to handle unpacking as well. (`for i,j in it`)
        target = for_node.target.id

        # Parse the template to AST.
        ast_template = LinenoStripper().visit(ast.parse(self.template, mode='exec'))

        # Extract the if statement.
        if_stmt: ast.If = ast_template.body[0].body[0]

        # Extract the inner function definition.
        inner_func: ast.FunctionDef = ast_template.body[0].body[1]

        # Variables are shared except for target
        shared = [name for name in self.list_locals(for_node.body) if name != target]

        # Replace the pass statement in the if body.
        if_stmt.body = self.replace(if_stmt.body, self.assign_shared(shared + list(self.collapse.keys())))

        # Replace the pass statement in the inner function body.
        nonlocals = []
        if len(shared) > 0:
            nonlocals = [ast.Nonlocal(names=[name for name in shared if name != target])]

        inner_func.body = self.replace(inner_func.body, nonlocals + node.body)
        return ast_template.body[0]
