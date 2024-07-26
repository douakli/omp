import ast


class Directive:

    def parse(self, node: ast.With) -> ast.With:
        """
        Convert the given With OpenMP construct with its implementation.
        """
        raise NotImplementedError()

    @staticmethod
    def replace(target: list[ast.AST], content: list[ast.AST], match: ast.AST = ast.Pass) -> list[ast.AST]:
        """
        Return a new list where the elements of content are where there is a Pass in target.
        """
        res = []
        for el in target:
            if isinstance(el, match):
                res.extend(content)
            else:
                res.append(el)
        return res

    @staticmethod
    def assign_shared(shared: list[str], value: ast.AST = None) -> ast.Assign:
        """
        Return a tuple assignment to the shared variables listed.
        """

        # Apparently, assigning an empty list to an empty tuple is valid python syntax.
        return [ast.Assign(
            targets=[ast.Tuple(elts=[ast.Name(id=name, ctx=ast.Store()) for name in shared],
                               ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.List(elts=[ast.Constant(value=value)], ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Constant(value=len(shared))
                )
            )]

    @staticmethod
    def list_locals(body: list[ast.AST]) -> list[str]:
        """
        Return a list of the local variables used in the given function body.
        """

        template = """\
def _omp_internal_inner_func():
    pass
"""

        ast_template = ast.parse(template, mode='exec')

        inner_func = ast_template.body[0]

        inner_func.body = Directive.replace(inner_func.body, body)

        # To find the list of local variables defined in the function, we need to compile it.
        globs, locs = dict(), dict()
        exec(compile(ast_template, '<OMP Parser>', mode='exec'), globs, locs)

        return locs['_omp_internal_inner_func'].__code__.co_varnames + locs['_omp_internal_inner_func'].__code__.co_cellvars


class OpenMP:

    """
    This class represents a call to an OpenMP directive.

    When used in omp-enabled user code, this class calls an OpenMP directive.

    When this class is used as a context manager, the with statement is replaced
    by the implementation of the construct.
    """

    # All the directives supported by the library. This variable is global.
    # To register a new directive, update omp.openmp.OpenMP.directives.
    directives: dict[str, Directive] = {}

    def __init__(self, directive: str = ''):
        self.directive = directive
        # TODO: Handle directives that are not constructs.

    def __enter__(self):
        # TODO: Check if we are in an omp-enabled context and decide what to do.
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def _parse_With(self, node: ast.With) -> ast.With:
        """
        This method is meant for internal use.

        Replace this OpenMP context manager by its implementation.
        """

        directive: str = self.directive.split()[0]

        if directive in OpenMP.directives:
            return OpenMP.directives[directive].parse(node)

        return node
