# encoding: utf-8


import torchvision.transforms as T
from data.transforms.auto_augment import AutoAugment,Cutout

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        if cfg.DATASETS.AUGMENTATION:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                AutoAugment(),
                Cutout(),
                T.ToTensor(),
                normalize_transform,
                # RandomErasing(probability=cfg.INPUT.PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                normalize_transform
            ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
