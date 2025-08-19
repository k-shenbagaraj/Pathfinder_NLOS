import os
import platform
import random
import time
from decimal import Decimal

import numpy as np
from loguru import logger
from prettytable import PrettyTable
from tqdm import tqdm
from fairscale.optim.oss import OSS
import torch
from torch import optim, nn, distributed
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader


# cudnn.benchmark = True


def seed_worker(worker_id):
    # print(torch.initial_seed())
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _set_seed(seed, deterministic=False):
    """
    seed manually to make runs reproducible
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option
        for CUDNN backend
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
