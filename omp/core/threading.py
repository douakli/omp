import threading

import omp.core.primitives
import omp


class Thread(threading.Thread):
    """
    Represents a thread managed by omp
    """

    def __init__(self, rank=None, team=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.team: Team = team
        self.icv = omp.core.primitives.InternalControlVariables(self)
        self.omp_parsing = False


class Team:

    """
    Represents a team of threads.
    """

    OMP_NUM_THREADS = 'OMP_NUM_THREADS'

    def __init__(self, size=None, *args, **kwargs):

        # The default team size is set by the OMP_NUM_THREADS environment variable.
        # If OMP_NUM_THREADS is not set, we use the number of available CPUs as the default size.
        if size is None:
            size = omp.get_max_threads()

        self.size = size
        self.threads = [Thread(i, self, *args, **kwargs) for i in range(self.size)]

        self.barrier = threading.Barrier(size)
        self.lock = threading.Lock()

        # ALERT: Access atomically.
        self.singleThread = None

        self.globalvars = {}

    def start(self):
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            thread.join()


def barrier():
    threading.current_thread().team.barrier.wait()


# Add our attributes to the main thread.
_mainThread = threading.current_thread()
_mainThread.rank = 0
_mainThread.team = Team(size=0)
_mainThread.team.size = 1
_mainThread.team.threads.append(_mainThread)
_mainThread.omp_parsing = False
