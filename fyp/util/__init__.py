import os
import random
from contextlib import contextmanager
from .note import *

# A little function to clear cuda cache. Put the import inside just in case we do not need torch, because torch import takes too long
def clear_cuda():
    import torch
    import gc
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            pass

def is_ipython():
    try:
        __IPYTHON__ #type: ignore
        return True
    except NameError:
        return False
