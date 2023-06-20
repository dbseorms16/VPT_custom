#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)
import torch.distributed as dist
import os

logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}

dist_url = 'env://'
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
    
dist.init_process_group("nccl", init_method=dist_url, world_size=world_size, rank=rank)
torch.cuda.set_device(local_rank)
dist.barrier()

def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        # from .datasets.tf_dataset import TFDataset
        # dataset = TFDataset(cfg, split)
        from .datasets.custom_dataset import CustomText
        from .datasets.augmentation import Augmentation
        dataset = CustomText(
        data_root='../SAM_customizing/data/Custom_data',
        is_training=True,
        load_memory=False,
        cfg=cfg,
        transform= Augmentation(size=1024, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
        
        
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
