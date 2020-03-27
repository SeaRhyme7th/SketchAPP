import time
from functools import wraps
import numpy as np
import torch
from torch.autograd import Variable

"""
refer to this repo:
    https://github.com/weixu000/DSH-pytorch
"""

def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


@timing
def compute_AP(cls_num_s, order_cls_num_p):
    """
    compute precision, recall and mAP from 330sketches
    all methods are based on [SHREC14-Sketch](https://www.itl.nist.gov/iad/vug/sharp/contest/2014/SBR/Evaluation.html)
    """

    Ns = range(1, len(order_cls_num_p) + 1)
    correct = (cls_num_s == order_cls_num_p).cumsum()
    P = correct / Ns
    AP = np.sum(P * correct) / sum(correct)

    return AP