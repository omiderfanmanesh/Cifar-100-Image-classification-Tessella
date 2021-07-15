# encoding: utf-8
import torch


# from .model import MusicClassificationCRNN
from torchvision.models.resnet import resnet18 as _resnet18


def build_model(cfg):
    is_pretrained = cfg.MODEL.PRE_TRAINED
    model = _resnet18(pretrained=is_pretrained)
    return model

