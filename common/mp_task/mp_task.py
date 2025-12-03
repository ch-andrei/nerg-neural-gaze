import multiprocessing as mp
import time
from datetime import datetime
from queue import Empty, Queue

from .util import split_list
from .atomic_counter import AtomicCounter


class MpTask(object):
    def __init__(self,
                 task_count: int,
                 task_function: callable,
                 task_pre_fuction: callable = None,
                 task_post_fuction: callable = None,
                 ):
        super(MpTask, self).__init__()
        self.task_count = task_count
        self.task_function = task_function
        self.task_pre_fuction = task_pre_fuction
        self.task_post_fuction = task_post_fuction
        self.task_counter = AtomicCounter()


def __process_task(queue, process_id, items, task: MpTask, task_args, task_kwargs, raise_exceptions=True):
    """

    :param queue:
    :param process_id:
    :param items:
    :param task:
    :param task_args:
    :param task_kwargs:
    :return:
    """
    pre_function_dict = None
    if task.task_pre_fuction is not None:
        pre_function_dict = task.task_pre_fuction()

    if pre_function_dict is not None and isinstance(pre_function_dict, dict):
        for key in pre_function_dict:
            task_kwargs[key] = pre_function_dict[key]

    for item in items:
        try:
            item_processed = task.task_function(item, *task_args, **task_kwargs)
            queue.put((item, item_processed))
        except Exception as e:
            if raise_exceptions:
                raise e
            else:
                print(f"[{process_id}] Processing raised exception: {e}")
                queue.put((item, e))
        task.task_counter.increment()
        print(f"[{process_id}][{datetime.now().strftime('%H:%M:%S')}]: {task.task_counter.value}/{task.task_count}")

    if task.task_post_fuction is not None:
        task.task_post_fuction()


def process_multithreaded(items: list, task_func: callable, task_args: list = None, task_kwargs: dict = None,
                          task_pre: callable = None, task_post: callable = None,
                          num_processes=1, aggressive=True, aggressive_period=1, raise_exceptions=False
                          ):
    """
    :param items: list of items to process
    :param task: function with signature f(item, *args, **kwargs); takes in and processes a single item
    :param task_args: list of arguments for task
    :param task_kwargs: dict of keyword arguments for task
    :param task_init:
    :param task_init_args:
    :param task_init_kwargs:
    :param num_processes:
    :param aggressive: when False, use process.join() to wait for the completion of tasks.
                        In some cases, .join() never returns, which results in a deadlock (program never completes).
                        If running the tasks hangs on .join(), try running with aggressive=True.
                        when True, busy wait on the processed item queue (check if all items are processed),
                        then simply forcibly stop the task processes.
    :param aggressive_period: time in seconds between busy wait checks on the processed item queue.
                        1 second by default.
    :return:
    """
    task_count = len(items)

    if task_args is None:
        task_args = list()

    if task_kwargs is None:
        task_kwargs = dict()

    task = MpTask(task_count, task_func, task_pre, task_post)
    num_processes = max(1, min(num_processes, task_count))  # limit number of processes by number of tasks
    items_processed = {}
    if 1 < num_processes:
        # set up multiprocessing
        queue = mp.Queue()  # multiprocessing queue

        items_per_process = split_list(items, num_processes)

        # spawn processes
        processes = []
        for process_id in range(num_processes):
            process = mp.Process(
                target=__process_task,
                args=(queue, process_id, items_per_process[process_id], task, task_args, task_kwargs),
                kwargs={"raise_exceptions": raise_exceptions}
            )
            process.start()
            processes.append(process)

        if aggressive:
            # busy wait, check mp.queue for items, try to get items until all tasks are processed
            tasks_processed = 0
            while tasks_processed < task_count:
                try:
                    item, item_processed = queue.get_nowait()
                    items_processed[item] = item_processed
                    tasks_processed += 1
                except Empty:
                    time.sleep(aggressive_period)

            for process_id in range(num_processes):
                processes[process_id].terminate()

        else:
            # normal scenario, processes that join correctly
            for process_id in range(num_processes):
                processes[process_id].join()

        del processes

    else:
        # single thread, no multiprocessing
        # run on current process
        process_id = 0
        queue = Queue()  # regular queue
        __process_task(queue, process_id, items, task, task_args, task_kwargs)

    # items already pulled out of the queue for aggressive multiprocessing
    if not (1 < num_processes and aggressive):
        for i in range(task_count):
            item, item_processed = queue.get()
            items_processed[item] = item_processed

    if 1 < num_processes:
        queue.close()
        queue.join_thread()

    return items_processed
