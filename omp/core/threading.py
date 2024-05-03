import os
import threading


class Thread(threading.Thread):
    """
    Represents a thread managed by omp
    """

    def __init__(self, rank=None, team=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.team: Team = team


class Team:

    """
    Represents a team of threads.
    """

    OMP_NUM_THREADS = 'OMP_NUM_THREADS'

    def __init__(self, size=None, *args, **kwargs):

        # The default team size is set by the OMP_NUM_THREADS environment variable.
        # If OMP_NUM_THREADS is not set, we use the number of available CPUs as the default size.
        if size is None:
            size = os.cpu_count() if self.OMP_NUM_THREADS not in os.environ else int(os.environ[self.OMP_NUM_THREADS])

        self.size = size
        self.threads = [Thread(i, self, *args, **kwargs) for i in range(self.size)]

    def start(self):
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            thread.join()
