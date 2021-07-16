from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR


def scheduler(optimizer):
    step_scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    lr_scheduler = LRScheduler(step_scheduler)
    return lr_scheduler
