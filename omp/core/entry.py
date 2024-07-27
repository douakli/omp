import omp.core.openmp
from omp.core.ast_tools import LinenoStripper

import ast
import copy
import inspect
import textwrap
import threading

from types import CodeType, FunctionType


class OpenMPTransformer(ast.NodeTransformer):
    """
    Recursively find the OpenMP constructs and replace them with their implementations.
    """

    def __init__(self, locs=None, globs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.locs = locs
        self.globs = globs

    def visit_With(self, node: ast.With) -> ast.With:
        # We need to make sure that this is an OpenMP construct.

        # The with statement should use only one context manager.
        if len(node.items) != 1:
            return self.generic_visit(node)

        # Bulletproofing.
        if not isinstance(node.items[0], ast.withitem):
            return self.generic_visit(node)

        # We are now sure we have a withitem.
        item: ast.withitem = node.items[0]

        # The context manager should be an OpenMP instanciation.
        # This means calling the constructor.

        if not isinstance(item.context_expr, ast.Call):
            return self.generic_visit(node)

        call: ast.Call = item.context_expr

        # Check that the function is referenced by Attributes of a Name, or a Name directly.
        ref = call.func
        while isinstance(ref, ast.Attribute):
            ref: ast.Attribute
            # Bulletproofing
            if not isinstance(ref.ctx, ast.Load):
                return self.generic_visit(node)
            ref = ref.value

        if not isinstance(ref, ast.Name):
            return self.generic_visit(node)

        name: ast.Name = ref

        # Bulletproofing.

        if not isinstance(name.ctx, ast.Load):
            return self.generic_visit(node)

        # In order to check that this call indeed an OpenMP instanciation,
        # we will evaluate the name being called in the function's definition namespace.

        expr: ast.Expression = ast.Expression(call.func)
        try:
            called = eval(compile(expr, filename='<OMP Parser>', mode='eval'), self.globs, self.locs)
        except NameError:
            # The name we are trying to evaluate could be undefined in the context.
            return self.generic_visit(node)

        if called is not omp.core.openmp.OpenMP:
            return self.generic_visit(node)

        # We are now sure this is an OpenMP construct. (Not necessarily a valid one.)
        # We run the found instanciation and run its logic on the found construct.
        instruction = eval(compile(ast.Expression(call), filename='<OMP Parser>', mode='eval'), self.globs, self.locs)

        # Parsing this node **after** its children allows us to know the exhaustive list of local variables that will be
        # involved in the inner function definitions.
        return instruction._parse_With(ast.fix_missing_locations(self.generic_visit(node)))


class EnableFunction(ast.NodeTransformer):
    """
    Transforms a enable-decorated function definition into an enabled function
    definition without the enable decorator.
    """

    def __init__(self, locs=None, globs=None, varnames=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.locs = locs
        self.globs = globs
        self.varnames = varnames
        if self.varnames is None:
            self.varnames = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        new = copy.deepcopy(node)
        # Remove the last decorator which should be ours if the user followed our documentation.
        new.decorator_list = new.decorator_list[:-1]
        # Inject a known name in the namespace for our library.
        new.body = [LinenoStripper().visit(ast.parse('import omp as _omp_internal', mode='exec').body[0])] + new.body

        # Run OpenMPTransformer until the result stabilizes.
        old_ver: str = ''
        while old_ver != ast.dump(new):
            old_ver = ast.dump(new)
            new = OpenMPTransformer(self.globs, self.locs).visit(ast.fix_missing_locations(new))

        return new


def enable(*args, **kwargs):
    """
    Enable OpenMP in the given block of code.

    Note: If your function has several decorators, this one should be the first to run. (i.e. closest to the function definition)

    Use as a decorator:
    ```
    @omp.enable
    def main():
        <omp-enabled code...>
    ```

    # Call your function
    main()
    """

    def decorator(function):
        threading.current_thread().omp_parsing = True
        # The user code calls `enable` which itself calls `decorator`.
        # We need to go back two stack frames.
        caller_frame = inspect.currentframe().f_back.f_back
        globs, locs = caller_frame.f_globals, caller_frame.f_locals

        # We need _omp_internal for our own compilations.
        locs.update({'_omp_internal': omp})

        # Retrieve the source code of the decorated function.
        src: str = textwrap.dedent(inspect.getsource(function))

        # Convert the source code to ast.
        src_ast: ast.Module = ast.parse(src, mode='exec')
        src_ast = ast.increment_lineno(src_ast, caller_frame.f_lineno - src_ast.body[0].lineno + 1)

        # Patch the source ast.
        # We need to make sure that each node has a line number.
        # Since the initial function was already compiled a first time, we can recover the
        # local variables the function uses from its code object.
        patched_ast = ast.fix_missing_locations(
            EnableFunction(globs, locs, function.__code__.co_varnames).visit(src_ast)
            )

        # ALERT: Remove this debug print. (Shows the final transformed source code.)
        print(ast.unparse(patched_ast))

        # Compile the patched ast.
        patched: CodeType = compile(patched_ast, filename=inspect.getsourcefile(function), mode='exec')

        # redefine the function in the initial context.
        # TODO: Allow enabling a function with closure (nested enabled function)
        #       Note: This would involve locating the function definition in the module's source code and recompile the outer function entirely.
        exec(patched, globs, locs)

        threading.current_thread().omp_parsing = False

        return locs[function.__name__]

    # Simple decorator.
    if len(args) == 1 and isinstance(args[0], FunctionType):
        return decorator(args[0])

    # TODO: Parametrized decorator support.
    # TODO: If statement support.
    return decorator
