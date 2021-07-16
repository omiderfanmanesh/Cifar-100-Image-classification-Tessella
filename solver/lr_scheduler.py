from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR


def scheduler(optimizer):
    step_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = LRScheduler(step_scheduler)
    return lr_scheduler
