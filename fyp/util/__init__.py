import os
import random
from contextlib import contextmanager
from .note import *
import gc

# A little function to clear cuda cache. Put the import inside just in case we do not need torch, because torch import takes too long
def clear_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def is_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
