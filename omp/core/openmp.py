import ast
import threading


class Clause:

    def __init__(self, directive: 'Directive', args: str):
        self.directive = directive
        self.args = args

        self.directive.clauses.append(self)


class Directive:

    def __init__(self, openMP: 'OpenMP'):
        self.openMP = openMP
        self.clauses: list(Clause) = []

        self.privates = set()
        self.nowait = False
        self.collapse = dict()

    def parse(self, node: ast.With) -> ast.With:
        """
        Convert the given With OpenMP construct with its implementation.
        """
        # raise NotImplementedError()
        pass

    def run(self):
        """
        Implementation of an OpenMP directive
        """
        #raise NotImplementedError()
        pass

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

    # @staticmethod
    def list_locals(self, body: list[ast.AST]) -> list[str]:
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

        return set(locs['_omp_internal_inner_func'].__code__.co_varnames + locs['_omp_internal_inner_func'].__code__.co_cellvars) - self.privates


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
    clauses: dict[str, Clause] = {}

    dir_impl: Directive = None

    @staticmethod
    def directive(name: str):
        """
        This decorator is meant for internal use.

        Register a new directive.
        """

        def decorator(cls):
            OpenMP.directives.update({name: cls})
        return decorator

    @staticmethod
    def clause(name: str, directives: list[str]):
        """
        This decorator is meant for internal use.

        Register a new clause.
        """
        def decorator(cls):
            OpenMP.clauses.update({name: (cls, directives)})
        return decorator

    def parse_clauses(self, clauses, directive):
        i = 0
        while i < len(clauses):
            if clauses[i] in ('(', ',', ' '):
                break
            i += 1

        clause = clauses[:i]
        args = ''

        if i < len(clauses):
            j = i + 1
            if clauses[i] == '(':
                while j < len(clauses):
                    if clauses[j] == ')':
                        break
                    j += 1
            args = clauses[i+1:j]
            self.parse_clauses(clauses[j+1:].strip(',').strip(), directive)

        if clause in self.clauses:
            cls, dirs = self.clauses[clause]
            if directive in dirs:
                cls(self.dir_impl, args)

    def __init__(self, arg: str = ''):
        words: list(str) = arg.split()

        directive: str = words[0] if words else arg

        # Allow for `parallel for` and future shortcut directives.
        for word in words[1:]:
            if f'{directive} {word}' not in self.directives:
                break
            directive = f'{directive} {word}'

        if directive in OpenMP.directives:
            self.dir_impl = OpenMP.directives[directive](self)
            self.clause_str = arg[len(directive):].strip()
            self.parse_clauses(arg[len(directive):].strip(), directive)

        if not threading.current_thread().omp_parsing and self.dir_impl is not None:
            self.dir_impl.run()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def _parse_With(self, node: ast.With) -> ast.With:
        """
        This method is meant for internal use.

        Replace this OpenMP context manager by its implementation.
        """

        if self.dir_impl is not None:
            return self.dir_impl.parse(node)

        return node
