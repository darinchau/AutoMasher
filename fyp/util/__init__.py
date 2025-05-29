import os
import random
from contextlib import contextmanager
from .note import *
from .url import *
from .load import *
import gc


def clear_cuda():
    # A little function to clear cuda cache. Put the import inside just in case we do not need torch, because torch import takes too long
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
