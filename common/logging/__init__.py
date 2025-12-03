from .logger import Logger, FileLogger
from ..timer import Timer

__global_logger = Logger()


def __tag(tag, *args):
    if tag:
        args = (f'[{tag}]',) + args
    return args


def log(*args, sep=" ", end="\n", tag=""):
    args = __tag(tag, *args)
    __global_logger(*args, sep=sep, end=end)


def log_warn(*args, sep=" ", end="\n", tag=""):
    args = __tag(tag, *args)
    log(*args, sep=sep, end=end, tag="WARNING")


class LogOnTaskComplete(object):
    def __init__(self, task_name, timer=True):
        self.task_name = task_name
        self.timer = None
        if timer:
            self.timer = Timer(task_name)

    def __enter__(self):
        log("Starting...", tag=self.task_name)
        if self.timer is not None:
            self.timer.__enter__()

    def __exit__(self, *args):
        if self.timer is not None:
            self.timer.__exit__(*args)
        log("Complete.", tag=self.task_name)

