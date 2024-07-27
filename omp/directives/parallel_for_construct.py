from omp.core.openmp import Directive, OpenMP
from omp.core.ast_tools import LinenoStripper

import ast


@OpenMP.directive('parallel for')
class ParallelForConstruct(Directive):

    """
    OpenMP parallel construct implementation.
    """

    @property
    def template(self):
        return f"""\
with _omp_internal.core.openmp.OpenMP("parallel"):
    with _omp_internal.core.openmp.OpenMP("for {self.openMP.clause_str}"):
        pass # Replaced by user code
        """

    def parse(self, node: ast.With) -> ast.With:
        # Parse the template to AST.
        ast_template = LinenoStripper().visit(ast.parse(self.template, mode='exec'))

        # Extract the inner with statement.
        with_stmt: ast.With = ast_template.body[0].body[0]

        # Replace the pass statement in the inner with body.
        with_stmt.body = self.replace(with_stmt.body, node.body)
        return ast_template.body[0]
