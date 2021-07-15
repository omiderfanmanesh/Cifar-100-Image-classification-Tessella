# encoding: utf-8

import numpy as np
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from data.dataset import CIFAR100
from data.transforms import build_transforms


def build_dataset(cfg, transforms):
    # load the dataset
    dataset = CIFAR100(
        root=cfg.DATASETS.ROOT, train=True,
        download=True, transform=transforms,
    )

    return dataset


def make_data_loader(cfg, inference=False, validation_size=0.10, shuffle=True):
    train_transformation = build_transforms(cfg, is_train=True)
    test_transformation = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH

    if inference:
        test_dataset = build_dataset(cfg, transforms=test_transformation)
        test_data_loader = data.DataLoader(
            test_dataset, batch_size=num_workers, shuffle=False, num_workers=num_workers
        )
        return test_data_loader

    train_dataset = build_dataset(cfg=cfg, transforms=train_transformation)
    validation_dataset = build_dataset(cfg=cfg, transforms=test_transformation)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))
    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    print(f"training sampler size : {len(train_sampler)}")
    print(f"validation sampler size : {len(valid_sampler)}")
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=batch_size,  num_workers=num_workers, sampler=train_sampler
    )

    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    print(f"training samples : {len(train_data_loader)}")
    print(f"validation samples : {len(validation_data_loader)}")
    return train_data_loader, validation_data_loader
