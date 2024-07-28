import omp
import os
import threading
from enum import Enum

import omp.core.threading


class Sched(Enum):
    static = 1
    dynamic = 2
    guided = 3
    auto = 4
    runtime = 5


class InternalControlVariables:

    num_procs_var = os.cpu_count()

    _OMP_NUM_THREADS = 'OMP_NUM_THREADS'
    _nthreads_var = num_procs_var if _OMP_NUM_THREADS not in os.environ else int(os.environ[_OMP_NUM_THREADS])

    @property
    def nthreads_var(self):
        return InternalControlVariables._nthreads_var

    @nthreads_var.setter
    def nthreads_var(self, value):
        InternalControlVariables._nthreads_var = value

    _OMP_NUM_TEAMS = 'OMP_NUM_TEAMS'
    _nteams_var = 0 if _OMP_NUM_TEAMS not in os.environ else int(os.environ[_OMP_NUM_TEAMS])

    @property
    def nteams_var(self):
        return InternalControlVariables._nteams_var

    @nteams_var.setter
    def nteams_var(self, value):
        InternalControlVariables._nteams_var = value

    _OMP_SCHEDULE = 'OMP_SCHEDULE'
    _run_sched_var = (Sched.dynamic, 1) if _OMP_SCHEDULE not in os.environ else (Sched[os.environ[_OMP_SCHEDULE]], 1)

    @property
    def run_sched_var(self):
        return InternalControlVariables._run_sched_var

    @run_sched_var.setter
    def run_sched_var(self, value):
        InternalControlVariables._run_sched_var = value

    def queue_size(self):
        return get_num_threads()*2

    def __init__(self, thread: 'omp.core.threading.Thread'):
        self.thread_num_var = thread.rank
        self.team_size_var = thread.team.size


def get_num_procs():
    return threading.current_thread().icv.numprocs_var


def set_num_threads(n: int):
    threading.current_thread().icv.nthreads_var = n


def get_max_threads():
    return threading.current_thread().icv.nthreads_var


def set_num_teams(n: int):
    threading.current_thread().icv.nteams_var = n


def get_max_teams():
    return threading.current_thread().icv.nteams_var


def get_thread_num():
    return threading.current_thread().icv.thread_num_var


def get_num_threads():
    return threading.current_thread().icv.team_size_var


def get_dynamic():
    return False


def set_schedule(kind: Sched, chunk=1):
    threading.current_thread().icv.run_sched_var = (kind, chunk)


def get_schedule():
    return threading.current_thread().icv.run_sched_var
