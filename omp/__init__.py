import omp.core as core
import omp.directives as directives
import omp.clauses as clauses

from omp.core.openmp import OpenMP
from omp.core.entry import enable

from threading import current_thread

# Avoid linter warnings for package shortcuts definitions.
core
directives
clauses

OpenMP
enable

get_num_procs = core.primitives.get_num_procs
set_num_threads = core.primitives.set_num_threads
get_max_threads = core.primitives.get_max_threads
set_num_teams = core.primitives.set_num_teams
get_max_teams = core.primitives.get_max_teams
get_thread_num = core.primitives.get_thread_num
get_num_threads = core.primitives.get_num_threads
get_dynamic = core.primitives.get_dynamic

current_thread().icv = core.primitives.InternalControlVariables(current_thread())
