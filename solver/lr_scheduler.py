from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR


def scheduler(optimizer):
    step_scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160])
    # lr_scheduler = GradualWarmupScheduler(
    #     optimizer,
    #     multiplier=2,
    #     total_epoch=3,
    #     after_scheduler=step_scheduler
    # )
    lr_scheduler = LRScheduler(step_scheduler)
    return lr_scheduler
