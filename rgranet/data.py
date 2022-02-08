import torch
from torchvision import datasets as D
from torchvision import transforms as T

DEFAULT_TRANSFORM = {
    "CIFAR10": {
        "test": T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "train": T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            #T.RandomAffine(degrees=15, translate=(.1,.1), scale=(.9 ,1.1), shear=10, resample=False, fillcolor=0),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    },
    "MNIST": T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ]),
}

SEED_TRAIN_VALID_SPLIT = 123

def CIFAR10(root, train=True, transform=None, target_transform=None, download=False):
    return D.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

def MNIST(root, train=True, transform=None, target_transform=None, download=False):
    return D.MNIST(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

def CIFAR10_DataLoaders(root, batch_size_train, batch_size_test, transform_train=None, transform_test=None, target_transform=None, validate_pct=0.0, manual_seed_valid_split=SEED_TRAIN_VALID_SPLIT, **kwargs):
    if transform_train is None:
        transform_train = DEFAULT_TRANSFORM["CIFAR10"]["train"]
    if transform_test is None:
        transform_test = DEFAULT_TRANSFORM["CIFAR10"]["test"]

    trainset = CIFAR10(root, train=True, transform=transform_train, target_transform=target_transform, download=True)
    validset = None
    if validate_pct is not None and validate_pct > 0.0:
        generator = torch.Generator.manual_seed(manual_seed_valid_split) if manual_seed_valid_split is not None else torch.default_generator
        trainset, validset = torch.utils.data.random_split(trainset, [len_train:=int(len(trainset) * (1.0 - validate_pct)), len(trainset) - len_train], generator=generator)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_train, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(
        CIFAR10(root, train=False, transform=transform_test, target_transform=target_transform, download=True),
        batch_size=batch_size_test, shuffle=False, **kwargs)
    validloader = torch.utils.data.DataLoader( 
        validset,
        batch_size=batch_size_test, shuffle=False, **kwargs
    ) if validset is not None else None
    return trainloader, testloader, validloader

def MNIST_DataLoaders(root, batch_size_train, batch_size_test, transform_train=None, transform_test=None, target_transform=None, validate_pct=0.0, manual_seed_valid_split=SEED_TRAIN_VALID_SPLIT, **kwargs):
    if transform_train is None:
        transform_train = DEFAULT_TRANSFORM["MNIST"]
    if transform_test is None:
        transform_test = DEFAULT_TRANSFORM["MNIST"]

    trainset = MNIST(root, train=True, transform=transform_train, target_transform=target_transform, download=True)
    validset = None
    if validate_pct is not None and validate_pct > 0.0:
        generator = torch.Generator.manual_seed(manual_seed_valid_split) if manual_seed_valid_split is not None else torch.default_generator
        trainset, validset = torch.utils.data.random_split(trainset, [len_train:=int(len(trainset) * (1.0 - validate_pct)), len(trainset) - len_train], generator=generator)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size_train, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(
        MNIST(root, train=False, transform=transform_test, target_transform=target_transform, download=True),
        batch_size=batch_size_test, shuffle=False, **kwargs)
    validloader = torch.utils.data.DataLoader( 
        validset,
        batch_size=batch_size_test, shuffle=False, **kwargs
    ) if validset is not None else None
    return trainloader, testloader, validloader