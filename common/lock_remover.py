import os
from pathlib import Path

from common.logging import log_warn

# AC Note: on windows, sometimes we run into deadlocks when/after building models with ninja.
# The fix is to remove the leftover "lock" files manually from pytorch cache.
# More info: https://github.com/eladrich/pixel2style2pixel/issues/80

__pycache_folder = "py312_cu126"  # this should be modified to fit local versions
__path_pycache = f"{Path.home()}/AppData/Local/torch_extensions/torch_extensions/Cache/{__pycache_folder}"


def check_lock_files(remove=False):
    for folder in os.listdir(__path_pycache):
        path2 = f"{__path_pycache}/{folder}"

        for file in os.listdir(path2):
            if file == "lock":
                log_warn(f"Found a lock file [{path2}]...")
                if remove:
                    log_warn(f"Lock file removed [{path2}].")
                    os.remove(f"{path2}/lock")


if __name__ == "__main__":
    check_lock_files(remove=True)
