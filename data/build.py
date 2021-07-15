# encoding: utf-8

from torch.utils import data

from data.dataset import CIFAR100
from data.transforms import build_transforms


def build_dataset(cfg, is_train=True, transforms=None):
    # load the dataset
    dataset = CIFAR100(
        root=cfg.DATASETS.ROOT, train=is_train,
        download=True, transform=transforms,
    )

    return dataset


def make_data_loader(cfg, inference=False, shuffle=True):
    train_transformation = build_transforms(cfg, is_train=True)
    test_transformation = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.TEST.IMS_PER_BATCH

    if inference:
        test_dataset = build_dataset(cfg, is_train=False, transforms=test_transformation)
        test_data_loader = data.DataLoader(
            test_dataset, batch_size=num_workers, shuffle=False, num_workers=num_workers
        )
        return test_data_loader

    train_dataset = build_dataset(cfg=cfg, transforms=train_transformation)

    train_data_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers
    )

    print(f"training samples : {len(train_data_loader)}")

    return train_data_loader
