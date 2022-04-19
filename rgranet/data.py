from enum import Enum
from pathlib import Path
from typing import Union

import torch
from torchvision import datasets as D
from torchvision import transforms as T

from .utils import dict_get_pop


class SupportedDataset(Enum):
    CIFAR10 = 0
    CIFAR100 = 1
    MNIST = 2
    IMAGENET = 3
    IMAGEDATASET = 4

class TransformType(Enum):
    BARE_MINIMUM = 0
    BASIC = 1

TRANSFORM_TYPE_PARSER = {
    "bare_minimum": TransformType.BARE_MINIMUM,
    "basic": TransformType.BASIC,
    None: None
}

NUM_CLASSES = {
    SupportedDataset.CIFAR10: 10,
    SupportedDataset.CIFAR100: 100,
    SupportedDataset.MNIST: 10,
    SupportedDataset.IMAGENET: 1000,
    SupportedDataset.IMAGEDATASET: None
}

def get_transforms(transform_type: TransformType, dataset: SupportedDataset):
    '''
    Returns minimum or basic transforms for each couple of dataset and transform_type
    '''
    if dataset == SupportedDataset.CIFAR10:
        bare_minimum = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        if transform_type.BARE_MINIMUM:
            return bare_minimum
        if transform_type.BASIC:
            return T.Compose([
                *bare_minimum.transforms,
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, 4)
            ])
    elif dataset == SupportedDataset.MNIST:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset == SupportedDataset.IMAGENET:
        bare_minimum = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if transform_type.BARE_MINIMUM:
            return bare_minimum
        if transform_type.BASIC:
            return T.Compose([
                *bare_minimum.transforms,
                T.RandomHorizontalFlip(),
                T.RandomCrop(224, 4)
            ])
    elif dataset == SupportedDataset.IMAGEDATASET:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    raise ValueError(f"Unsupported transform type {transform_type} or dataset type {dataset}")

def get_dataset(dataset:SupportedDataset, root:Union[str,Path], train:bool, transform=TransformType.BASIC, target_transform=None):
    '''
    Returns a torch dataset.

    Params:
        dataset: a SupportedDataset
        root: the root where the dataset is preset or is to be downloaded. Not all datasets (e.g. ImageNet) support downloading
        train: flag indicating whether to load the train or the test or the validation dataset (NB dataset dependent!)
        transform: either a TransformType or a functional to apply to the dataset while sampling or None (default TransformType.BASIC). If None, no transformation is applied
        target_transform: a function to apply to the target of the dataset (not required)

    Returns:
        a torch dataset
    '''
    if isinstance(transform, TransformType):
        transform = get_transforms(transform, dataset)
    if dataset == SupportedDataset.CIFAR10:
        return D.CIFAR10(root, train, transform, target_transform, download=True)
    elif dataset == SupportedDataset.CIFAR100:
        return D.CIFAR100(root, train, transform, target_transform, download=True)
    elif dataset == SupportedDataset.MNIST:
        return D.MNIST(root, train, transform, target_transform, download=True)
    elif dataset == SupportedDataset.IMAGENET:
        return D.ImageNet(root, split="train" if train else "val", transform=transform, target_transform=target_transform, download=False)
    elif dataset == SupportedDataset.IMAGEDATASET:
        return D.ImageFolder(root, transform=transform, target_transform=target_transform)
    raise ValueError(f"Unsupported dataset type {dataset}")
    

def get_torch_generator(
    manual_seed:int
):
    if manual_seed is not None: 
        return torch.Generator().manual_seed(manual_seed)
    return torch.default_generator

def get_dataloaders(
    dataset_name:SupportedDataset,
    dataset_root:Union[str, Path],
    batch_train:int,
    batch_test:int,
    batch_valid:int=None,
    pct_valid:float=0.0,
    transform_train=None,
    transform_test=None,
    target_transform=None,
    manual_seed_valid_split:int=None,
    manual_seed_trainloader:int=None,
    imagefolder_roots:dict=None,
    distributed:bool=False,
    distributed_world_size:int=None,
    distributed_rank:int=None,
    
    **kwargs
    ):
    '''
    Generic DataLoaders constructor for this project.
    Params:
        dataset_name: one of SupportedDataset
        dataset_root: root directory for the dataset
        batch_train: batch size for training
        batch_test: batch size for testing
        batch_valid: batch size for validation (not required. If pct_valid>0.0, this is same as batch_test)
        pct_valid: percentage of the dataset to use for validation (default 0.0 → no validation). Overridden if imagefolder_roots has a 'val' key
        transform_train: transforms to apply to the training set (default None → normalize data)
        transform_test: transforms to apply to the testing set. NB same applies to validation set (default None → normalize data)
        target_transform: transforms to apply to the targets (not required)
        manual_seed_valid_split: seed for the random split of the dataset (default None → use torch default generator)
        manual_seed_trainloader: seed for the random split of the dataset (default None → use torch default generator)
        imagefolder_roots: dictionary of imagefolder roots for keys 'test' and 'val' (not required)
        distributed: flag to use DistributedSampler in DataLoader construction (use in case of DistributedDataParallel model)
        distributed_world_size: number of processes in the distributed training (required if distributed)
        distributed_rank: rank of the process in the distributed training (required if distributed)
        **kwargs: additional arguments to pass to the DataLoader constructor

    Returns:
        trainloader, testloader, validloader (None if pct_valid=0.0 or None)
    
    '''
    assert pct_valid is None or (pct_valid >= 0.0 and pct_valid < 1.0), f"pct_valid must be between 0.0 and 1.0 (not included), but is {pct_valid}"

    drop_last = dict_get_pop(kwargs, 'drop_last', distributed)
    if distributed:
        assert drop_last is False, f"Cannot use drop_last=False in case of distributed training. Please fix."

    if isinstance(transform_train, str):
        transform_train = TRANSFORM_TYPE_PARSER[transform_train.lower()]
    if isinstance(transform_test, str):
        transform_test = TRANSFORM_TYPE_PARSER[transform_test.lower()]

    test_root = val_root = dataset_root
    if imagefolder_roots is not None:
        assert imagefolder_roots.get("test") is not None or imagefolder_roots.get("val") is not None, f"imagefolder_roots must contain at least one of 'test' and 'val' keys"
        test_root = imagefolder_roots.get("test")
        val_root = imagefolder_roots.get("val")

    trainset = get_dataset(dataset_name, dataset_root, train=True, transform=transform_train, target_transform=target_transform)

    if test_root is not None:
        testset = get_dataset(dataset_name, test_root, train=False, transform=transform_test, target_transform=target_transform)
        if distributed:
            assert (rem:=len(testset) % batch_test) == 0, f"For distributed training, the testset must be splittable in batches of size exactly {batch_test}. Current size is {len(testset)} with a remainder of {rem}. Please adjust the batch_test parameter"
    else:
        # covers the case of ImageFolder with no test set
        testset = None

    validset = None
    if dataset_name == SupportedDataset.IMAGEDATASET and val_root is not None:
        validset = get_dataset(dataset_name, val_root, train=False, transform=transform_test, target_transform=target_transform)
        if distributed:
            assert (rem:=len(validset) % batch_valid) == 0, f"For distributed training, the validation set must be splittable in batches of size exactly {batch_valid}. Current size is {len(validset)} with a remainder of {rem}. Please adjust the batch_valid parameter"
    
    # validset is None requested for the case of ImageFolder with no validation set but pct_valid > 0.0
    validate = (pct_valid is not None and pct_valid > 0.0 and validset is None)
    if validate:
        generator = get_torch_generator(manual_seed_valid_split) 
        trainset, validset = torch.utils.data.random_split(trainset, [len_train:=int(len(trainset) * (1.0 - pct_valid)), len(trainset) - len_train], generator=generator)
    
    trainsampler = testsampler = validsampler = None
    if distributed:
        trainsampler = torch.utils.data.distributed.DistributedSampler(
            trainset, num_replicas=distributed_world_size, rank=distributed_rank, shuffle=True, seed=manual_seed_trainloader
        )
        testsampler = torch.utils.data.distributed.DistributedSampler(
            testset, num_replicas=distributed_world_size, rank=distributed_rank, shuffle=False, seed=manual_seed_trainloader
        )
        if validset is not None:
            validsampler = torch.utils.data.distributed.DistributedSampler(validset, num_replicas=distributed_world_size, rank=distributed_rank, shuffle=False, seed=manual_seed_trainloader)
    
    shuffle_train = (not distributed)
    trainloader_generator = get_torch_generator(manual_seed_trainloader if (shuffle_train or not distributed) else None)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_train, shuffle=shuffle_train, generator=trainloader_generator, sampler=trainsampler, pin_memory=True, drop_last=drop_last, **kwargs)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_test, shuffle=False, sampler=testsampler, pin_memory=True, drop_last=drop_last, **kwargs)
    validloader = torch.utils.data.DataLoader( 
        validset, batch_size=batch_valid, shuffle=False, sampler=validsampler, pin_memory=True, drop_last=drop_last, **kwargs) if validset is not None else None
    return trainloader, testloader, validloader

