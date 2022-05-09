import argparse
import torch
from ast import arg
import parse_config
from main import get_data, get_model
from rgranet import model
from rgranet.data import NUM_CLASSES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config = parse_config.parse_config(args.config_path)

    _, testloader = get_data(config, use_ffcv=False)

    is_distributed = False # hasattr(config, "distributed") and config.distributed is not None

    assert hasattr(config, "eval"), f"For testing, the config file needs to have an 'eval' section."

    net_hyperparameters = vars(config.net_hyperparameters) if hasattr(config, "net_hyperparameters") else {}
    net_module = get_model(config.net, num_classes=NUM_CLASSES[config.data.name], **net_hyperparameters)

    mask_kwargs = vars(config.train.pruning.hyperparameters)
    if (pr_sched_class:=(vars(config.train.pruning).get("scheduler"))) is not None:
        mask_kwargs.pruning_rate_schedule = pr_sched_class
        if (pr_sched_hyp:=pr_sched_class.get("hyperparameters")) is not None:
            mask_kwargs.scheduling_kwargs = pr_sched_hyp
    mask_kwargs["tot_num_pruning_ite"] = 1

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

    params = torch.load(config.eval.params_path)
    net.load_trained_model(params)
    print("Loaded train model")

    net.evaluate(
        testloader=testloader,
        eval_loss=True
    )

if __name__ == "__main__":
    main()