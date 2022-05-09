import math

def cyclical_lr_determine_up_and_down_size(config, num_ite):
    ratio = config.train.optim.scheduler.hyperparameters.up_to_down_ratio
    inv_ratio = 1 / ratio
    config.train.optim.scheduler.hyperparameters.step_size_up = math.ceil(ratio * num_ite / (ratio + 1))
    config.train.optim.scheduler.hyperparameters.step_size_down = math.floor(num_ite * inv_ratio / (inv_ratio + 1))
    del config.train.optim.scheduler.hyperparameters.up_to_down_ratio

def cyclical_lr_determine_total_steps(config, num_epochs_without_burnnout, num_ite_per_epoch):
    if config.train.optim.scheduler._class.__name__ == "CyclicLRWithBurnout":
        if not hasattr(config.train.optim.scheduler.hyperparameters, "total_steps"):
            total_steps = num_epochs_without_burnnout * num_ite_per_epoch
            config.train.optim.scheduler.hyperparameters.total_steps = total_steps

def determine_base_lr(config):
    if not hasattr(config.train.optim.hyperparameters, "lr"):
        if hasattr(config.train.optim.scheduler.hyperparameters, "base_lr"):
            config.train.optim.hyperparameters.lr = config.train.optim.scheduler.hyperparameters.base_lr
        else:
            raise ValueError("Either 'lr' (in train->optim->hyperparameters) or 'base_lr' (in train->optim->scheduler->hyperparameters) must be specified in the config file.")
