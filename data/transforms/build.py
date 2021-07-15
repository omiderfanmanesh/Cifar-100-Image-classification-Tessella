# encoding: utf-8


import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        if cfg.DATASETS.AUGMENTATION:
            transform = T.Compose([
                T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN,
                                    scale=(cfg.INPUT.MIN_SCALE_TRAIN, cfg.INPUT.MAX_SCALE_TRAIN)),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.ToTensor(),
                normalize_transform
            ])
        else:
            transform = T.Compose([
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
