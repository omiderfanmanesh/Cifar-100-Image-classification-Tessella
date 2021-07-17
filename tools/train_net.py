# encoding: utf-8


import os
import sys
from os import mkdir,makedirs

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer,scheduler as sch

from utils.logger import setup_logger

import random

SEED = 2021
random.seed(SEED)

import torch
import torch.nn as nn

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.empty_cache()
import numpy as np

np.random.seed(SEED)


def train(cfg):
    model = build_model(cfg)

    optimizer = make_optimizer(cfg=cfg,
                               model_params=model.parameters())
    scheduler = sch(optimizer)

    arguments = {}

    train_loader = make_data_loader(cfg)

    criterion = nn.CrossEntropyLoss()

    # check_pointers = torch.load('../outputs/first/check_pointers/CIFAR100_checkpoint_468750.pt')
    # model.load_state_dict(check_pointers['model'])

    do_train(
        cfg,
        model,
        train_loader,
        None,
        optimizer,
        scheduler,
        criterion
    )


def main():
    num_gpus = 1

    output_dir = cfg.DIR.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        makedirs(output_dir)

    best_models = cfg.DIR.BEST_MODEL
    if best_models and not os.path.exists(best_models):
        makedirs(best_models)

    tensorboard_log = cfg.DIR.TENSORBOARD_LOG
    if tensorboard_log and not os.path.exists(tensorboard_log):
        makedirs(tensorboard_log)

    final_model = cfg.DIR.FINAL_MODEL
    if final_model and not os.path.exists(final_model):
        makedirs(final_model)

    root = cfg.DATASETS.ROOT
    if root and not os.path.exists(root):
        makedirs(root)

    logger = setup_logger("template_model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    train(cfg)


if __name__ == '__main__':
    main()
