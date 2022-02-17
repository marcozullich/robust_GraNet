from operator import is_
import os
import timm
import torch
import random

from rgranet import architectures 
from rgranet import data
from rgranet import lr_schedulers
from rgranet import pruning_mask as msk
from rgranet import pruning_rate_schedule as prs
from rgranet import model
from rgranet.utils import yaml_load, coalesce, is_int

def parse_env(string):
    return os.path.expandvars(string)


def parse_path(path):
    return os.path.expanduser(parse_env(path))

def parse_str(string):
    if string.strip() == "None":
        return None
    return string

def parse_config_none(config):
    for k, v in config.items():
        if isinstance(v, dict):
            parse_config_none(v)
        elif isinstance(v, str):
            config[k] = parse_str(v)

def parse_net(config):
    if config["net"] == "FCN4":
        config["net_class"] = lambda **kwargs: architectures.FCN4(**kwargs)
    elif config["net"] == "PreActResNet18":
        config["net_class"] = lambda num_classes, **kwargs: architectures.PreActResNet18(num_classes=num_classes, **kwargs)
    else:
        config["net_class"] = lambda num_classes, **kwargs: timm.create_model(config["net"], pretrained=False, num_classes=num_classes, **kwargs)

def parse_datasets(config):
    parser = {
        "mnist": data.MNIST_DataLoaders,
        "cifar10" : data.CIFAR10_DataLoaders,
    }
    config["data"]["dataset_loader"] = parser[config["data"]["dataset"].lower()]

def parse_loss(config):
    parser = {
        "Cross Entropy": torch.nn.CrossEntropyLoss,
    }
    config["train"]["loss"] = parser[config["train"]["loss"]]

def parse_optim(config):
    parser = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
    }
    config["train"]["optim"]["class"] = parser[config["train"]["optim"]["class"].upper()]

def parse_lr_scheduler(config):
    parser = {
        "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
        "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
        "CyclicLRWithBurnout": lr_schedulers.CyclicLRWithBurnout
    }
    config["train"]["optim"]["scheduler"]["class"] = parser[config["train"]["optim"]["scheduler"]["class"]]

def parse_milestone(config):
    parser = {
        "epoch": model.TrainingMilestone.END_EPOCH,
        "iteration": model.TrainingMilestone.END_ITERATION,
    }

    config["train"]["optim"]["scheduler"]["update_time"] = parser[config["train"]["optim"]["scheduler"]["update_time"].lower()]
    config["train"]["checkpoint_save_time"] = parser[config["train"]["checkpoint_save_time"].lower()]

def parse_mask(config):
    parser = {
        "None": msk.NoMask,
        "NoMask": msk.NoMask,
        "LMMask": msk.LMMask,
        "GraNetMask": msk.GraNetMask,
        "RGraNetMask": msk.RGraNetMask,
        "GradualPruningMask": msk.GradualPruningMask,
    }
    config["train"]["pruning"]["mask_class"] = parser[config["train"]["pruning"]["mask_class"]]

def parse_pr_scheduler(config):
    if config["train"]["pruning"].get("scheduler") is not None:
        parser = {
            "None": prs.NoPruningRateScheduling,
            "NoPruningRateScheduling": prs.NoPruningRateScheduling,
            "PruningRateCosineScheduling": prs.PruningRateCosineScheduling,
            "PruningRateLinearScheduling": prs.PruningRateLinearScheduling,
            "PruningRateCubicScheduling": prs.PruningRateCubicScheduling,
            "PruningRateCubicSchedulingWithRegrowth": prs.PruningRateCubicSchedulingWithRegrowth,
        }
        config["train"]["pruning"]["scheduler"]["class"]= parser[config["train"]["pruning"]["scheduler"]["class"]]

def parse_config(config_path):
    config = yaml_load(config_path)
    
    # parse_config_none(config)
    parse_net(config)
    parse_datasets(config)
    parse_loss(config)
    parse_optim(config)
    parse_lr_scheduler(config)
    parse_milestone(config)
    parse_mask(config)
    parse_pr_scheduler(config)
    config["data"]["hyperparameters"]["root"] = parse_path(config["data"]["hyperparameters"]["root"])
    config["util"]["telegram_config_name"] = parse_path(config["util"]["telegram_config_name"]) if config["util"]["telegram_config_name"] is not None else None

    if config.get("num_run") is not None:
        if config["num_run"].strip().startswith("$"):
            if config["num_run"] == "" or (not is_int(config["num_run"])): # if env variable does not exist, generate random hash
                config["num_run"] = int(random.random()*1e6)
        filename, ext = os.path.splitext(config["train"]["final_model_save_path"])
        config["train"]["final_model_save_path"] = f"{filename}_{config['num_run']}{ext}"

    return config
