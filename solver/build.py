from .optimizer import OptimizerFactory


def make_optimizer(cfg, model_params):
    op = OptimizerFactory(cfg=cfg, model_params=model_params, opt=cfg.MODEL.OPTIMIZER)
    optimizer = op.get_opt()
    return optimizer
