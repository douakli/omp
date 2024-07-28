import omp
from omp.core.openmp import Directive, OpenMP
from omp.core.ast_tools import LinenoStripper
from omp.core.primitives import Sched

import ast
import random
import queue
import time
import threading


def generator_static(it, nonce):
    """
    When called within a thread of a team, yields the iterations for the current thread.

    Overall, when all the threads of the team call this generator, all the elements of the iterator are yielded.
    """
    for i, el in enumerate(it):
        if i % omp.get_num_threads() == omp.get_thread_num():
            yield el


class EndOfQueue:
    pass


@OpenMP.directive('for')
class ForConstruct(Directive):

    schedule = Sched.runtime

    """
    OpenMP for construct implementation.
    """

    @property
    def template(self):

        reduction_vars, reduction_operators = list(zip(*self.reduction.items())) if self.reduction else ([], [])
        nonce = random.randint(0, 100000)

        return f"""\
with _omp_internal.core.openmp.OpenMP():
    if False:
        pass # Replaced by shared variables declarations
    def _omp_internal_inner_func{nonce}({','.join(reduction_vars)}):
        pass # Replaced by user code
        {'#' if self.nowait else ''}_omp_internal.core.openmp.OpenMP("barrier")
        {'' if reduction_vars else '#'}return ({','.join(reduction_vars)},)
    def _omp_internal_inner_func_protect{nonce}():
        {'' if reduction_vars else '#'}nonlocal {','.join(reduction_vars)}
        _omp_internal_retval = _omp_internal_inner_func{nonce}({','.join(reduction_vars)})
        {'' if reduction_vars else '#'}with _omp_internal.core.openmp.OpenMP('critical'):
            {'' if reduction_vars else '#'}({','.join(reduction_vars)},) = [_omp_internal.clauses.reduction.operators[_omp_internal_operator](_omp_internal_var, _omp_internal_val) for _omp_internal_operator, _omp_internal_var, _omp_internal_val in zip({reduction_operators}, ({','.join(reduction_vars)},), _omp_internal_retval)]
    _omp_internal_inner_func_protect{nonce}()
    """

    @staticmethod
    def define_generators():

        if hasattr(omp.directives.for_construct, 'dynamic'):
            return

        @omp.enable
        def generator_dynamic(it, nonce):
            with OpenMP("single"):
                threading.current_thread().team.globalvars[f'_omp_interal_for_q{nonce}'] = queue.Queue()
            q = threading.current_thread().team.globalvars[f'_omp_interal_for_q{nonce}']
            with OpenMP("single nowait"):
                for el in it:
                    q.put(el)
                for _ in range(omp.get_num_threads()):
                    q.put(EndOfQueue)
            while True:
                el = q.get()
                if el is EndOfQueue:
                    break
                yield el
            OpenMP("barrier")
            with OpenMP("single nowait"):
                del threading.current_thread().team.globalvars[f'_omp_interal_for_q{nonce}']

        omp.directives.for_construct.generator_dynamic = generator_dynamic
        omp.directives.for_construct.generator_guided = omp.directives.for_construct.generator_dynamic
        omp.directives.for_construct.generator_auto = omp.directives.for_construct.generator_dynamic

    def parse(self, node: ast.With):

        self.define_generators()

        for_node: ast.For = node.body[0]

        schedule = self.schedule

        if schedule == Sched.runtime:
            schedule = omp.get_schedule()

        # Wrap the loop iterator in our thread-distributing generator.
        for_node.iter = ast.Call(LinenoStripper().visit(ast.parse(f'_omp_internal.directives.for_construct.generator_{schedule.name}')).body[0].value, args=[for_node.iter, ast.Constant(value=time.time())], keywords=[])

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
        if_stmt.body = self.replace(if_stmt.body, self.assign_shared(shared + list(self.reduction.keys())))

        # Replace the pass statement in the inner function body.
        nonlocals = []
        if len(shared) > 0:
            nonlocals = [ast.Nonlocal(names=[name for name in shared if name != target])]

        inner_func.body = self.replace(inner_func.body, nonlocals + node.body)
        return ast_template.body[0]
