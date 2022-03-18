from operator import is_
import os
import timm
import torch
import random

from rgranet import architectures 
from rgranet import data
from rgranet import lr_schedulers
from rgranet import pruning_mask as msk
from rgranet.pruning_mask import GradientsAccumulationMethod
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

def parse_clip_grad_norm(config):
    if config["train"].get("clip_grad_norm") is not None and config["train"].get("clip_grad_norm_before_epoch") is not None:
        raise ValueError("clip_grad_norm and clip_grad_norm_before_epoch cannot be set at the same time")

    if config["train"].get("clip_grad_norm"):
        clip_grad_norm = config["train"].pop("clip_grad_norm")
        config["train"]["clip"]["clip_grad_norm_before_epoch"] = config["train"]["epochs"] + config["train"]["burnout_epochs"] + 1 if clip_grad_norm else 0

    

def parse_grad_accumul(config):
    if config["train"]["pruning"]["hyperparameters"].get("accumulate_gradients_before_regrowth") is not None and config["train"]["pruning"]["hyperparameters"].get("gradients_accumulation_method") is not None:
        raise ValueError("Cannot specify, in configuration, both accumulate_gradients_before_regrowth and gradients_accumulation_method")

    if config["train"]["pruning"]["hyperparameters"].get("accumulate_gradients_before_regrowth") is not None:
        grad_before_regrowth = config["train"]["pruning"]["hyperparameters"].pop("accumulate_gradients_before_regrowth")
        if grad_before_regrowth:
            config["train"]["pruning"]["hyperparameters"]["gradients_accumulation_method"] = GradientsAccumulationMethod.BETWEEN_PRUNE_AND_REGROWTH
        else:
            config["train"]["pruning"]["hyperparameters"]["gradients_accumulation_method"] = GradientsAccumulationMethod.NEVER

    elif config["train"]["pruning"]["hyperparameters"].get("gradients_accumulation_method") is not None:
        grad_accum_method = config["train"]["pruning"]["hyperparameters"]["gradients_accumulation_method"].lower()
        parser = {
            "never": GradientsAccumulationMethod.NEVER,
            "always": GradientsAccumulationMethod.ALWAYS,
            "between_prune_and_regrowth": GradientsAccumulationMethod.BETWEEN_PRUNE_AND_REGROWTH,
        }
        config["train"]["pruning"]["hyperparameters"]["gradients_accumulation_method"] = parser[grad_accum_method]


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

def config_savefile(config):
    if config.get("num_run") is None:
        config["num_run"] = int(random.random()*1e6)

    filename, ext = os.path.splitext(config["train"]["final_model_save_path"])
    config["train"]["final_model_save_path"] = f"{filename}_{config['num_run']}{ext}"

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
    parse_clip_grad_norm(config)
    parse_grad_accumul(config)
    parse_pr_scheduler(config)
    config_savefile(config)
    config["data"]["hyperparameters"]["root"] = parse_path(config["data"]["hyperparameters"]["root"])
    config["util"]["telegram_config_name"] = parse_path(config["util"]["telegram_config_name"]) if config["util"]["telegram_config_name"] is not None else None

        

    return config
