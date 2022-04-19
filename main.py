from operator import is_
import torch
import argparse
import math
import os
from pprint import pprint as pretty_print

import parse_config
import main_cyclical_lr as cyc
import distributed
from rgranet import model
from rgranet.utils import make_subdirectory
from rgranet.pruning_mask import NoMask
from rgranet.data import NUM_CLASSES, get_dataloaders
from rgranet.architectures import get_model



def determine_epoch_and_ite_start(epoch_start_from_checkpoint, ite_start_from_checkpoint, num_epochs, num_ite):
    if epoch_start_from_checkpoint == num_epochs - 1 and ite_start_from_checkpoint == num_ite - 1:
        raise ValueError("The loaded checkpoint refers to a completed training.")
    elif ite_start_from_checkpoint == num_ite - 1:
        ite_start_from_checkpoint = 0
        epoch_start_from_checkpoint += 1
    else:
        ite_start_from_checkpoint += 1
    return epoch_start_from_checkpoint, ite_start_from_checkpoint


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()
    return parse_config.parse_config(args.config_path)

def get_data(config):
    is_distributed = hasattr(config, "distributed") and config.distributed is not None
    world_size = rank = None
    if is_distributed:
        world_size = config.distributed.world_size
        rank = config.distributed.rank

    trainloader, testloader, _ = get_dataloaders(config.data.name, **vars(config.data.hyperparameters), distributed=is_distributed, distributed_world_size=world_size, distributed_rank=rank)

    return trainloader, testloader

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    config = get_config()
    is_distributed = hasattr(config, "distributed") 

    if is_distributed:
        distributed.handle_slurm(config)
    else:
        set_up_training(config=config)

def set_up_training(gpu=None, config=None):
    if config is None:
        config = get_config()

    is_distributed = hasattr(config, "distributed") and config.distributed is not None

    print("\t\tConfiguration:")
    pretty_print(config)

    if is_distributed:
        slurm_config = config.distributed
        distributed.init_dist_gpu(slurm_config)
    
    print(f"save file {config.train.final_model_save_path}")

    trainloader, testloader = get_data(config)

    if hasattr(config, "global_seed"):
        set_seeds(config.global_seed)

    tot_training_ite_without_burnout = len(trainloader) * config.train.epochs

    if config.train.optim.scheduler._class.__name__ in  ("CyclicLR", "CyclicLRWithBurnout"):
        cyc.cyclical_lr_determine_up_and_down_size(config, tot_training_ite_without_burnout)
        cyc.determine_base_lr(config)
        cyc.cyclical_lr_determine_total_steps(config, config.train.epochs, len(trainloader))

    net_hyperparamters = vars(config.net_hyperparameters) if hasattr(config, "net_hyperparameters") else {}
    net_module = get_model(config.net, num_classes=NUM_CLASSES[config.data.name], **net_hyperparamters)

    make_subdirectory(config.train.checkpoint_path)
    make_subdirectory(config.train.final_model_save_path)
    

    if config.train.pruning.mask_class != NoMask:
        if not hasattr(config.train.pruning,"scheduler"):
            if not hasattr(config.train.pruning.hyperparameters, "tot_num_pruning_ite"):
                config.train.pruning.hyperparameters.tot_num_pruning_ite = tot_training_ite_without_burnout // config.train.pruning.hyperparameters.pruning_frequency
        else:
            if not hasattr(config.train.pruning.scheduler.hyperparameters, "tot_num_pruning_ite"):
                config.train.pruning.scheduler.hyperparameters.tot_num_pruning_ite = tot_training_ite_without_burnout // config.train.pruning.scheduler.hyperparameters.pruning_frequency
            # config.train.pruning.scheduler.pruning_frequency = config.train.pruning.hyperparameters.pruning_frequency


    mask_kwargs = vars(config.train.pruning.hyperparameters)
    if (pr_sched_class:=(vars(config.train.pruning).get("scheduler"))) is not None:
        mask_kwargs.pruning_rate_schedule = pr_sched_class
        if (pr_sched_hyp:=pr_sched_class.get("hyperparameters")) is not None:
            mask_kwargs.scheduling_kwargs = pr_sched_hyp

    net = model.Model(
        module=net_module,
        optimizer_class=config.train.optim._class,
        optimizer_kwargs=config.train.optim.hyperparameters,
        lr_scheduler_class=config.train.optim.scheduler._class,
        lr_scheduler_kwargs=config.train.optim.scheduler.hyperparameters,
        lr_scheduler_update_time=config.train.optim.scheduler.update_time,
        loss_fn=config.train.loss(),
        mask=None,
        mask_class=config.train.pruning.mask_class,
        mask_kwargs=mask_kwargs,
        # name=config.net,
        distributed_device=config.distributed.gpu if is_distributed else None,
        is_main_device=(is_distributed and config.distributed.main) or (not is_distributed)
    )


    epoch_start = ite_start = 0
    if config.train.resume_training:
        if os.path.isfile(check_path:=config.train.checkpoint_path):
            epoch_start, ite_start = net.load_checkpoint(torch.load(check_path))
            epoch_start, ite_start = determine_epoch_and_ite_start(epoch_start, ite_start, config.train.epochs + config.train.burnout_epochs, len(trainloader))


    net.train_model(
        trainloader=trainloader,
        num_epochs=config.train.epochs,
        burnout_epochs=config.train.burnout_epochs,
        checkpoint_path=config.train.checkpoint_path,
        checkpoint_save_time=config.train.checkpoint_save_time,
        final_model_path=config.train.final_model_save_path,
        epoch_start=epoch_start,
        ite_start=ite_start,
        clip_grad_norm_before_epoch=config.train.clip_grad_norm_before_epoch,
    )

    net.evaluate(
        testloader=testloader,
        eval_loss=True,
    )

if __name__ == "__main__":
    main()
