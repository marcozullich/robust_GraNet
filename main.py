import torch
import argparse
import math
import os

import parse_config
import main_cyclical_lr as cyc
from rgranet.model import Model
from rgranet.utils import coalesce, make_subdirectory

NUM_CLASSES = {
    "cifar10": 10,
    "mnist": 10,
    "imagenet1k": 1000
}

def set_telegram_tokens(use_telegram, telegram_tokens_folder):
    if use_telegram:
        if telegram_tokens_folder is None:
            raise ValueError("Please specify the telegram tokens folder.")
        else:
            if os.path.isdir(telegram_tokens_folder):
                if os.path.isfile(chatid_path:=os.path.join(telegram_tokens_folder, "chatid.txt")) and os.path.isfile(token_path:=os.path.join(telegram_tokens_folder, "token.txt")):
                    return token_path, chatid_path
                else:
                    raise ValueError(f"Could not find chatid.txt and token.txt in {telegram_tokens_folder}")
            else:
                raise ValueError(f"{telegram_tokens_folder} is not a directory. Please DO NOT specify --use_telegram if you do not have telegram tokens.")
    else:
        return None, None


def determine_epoch_and_ite_start(epoch_start_from_checkpoint, ite_start_from_checkpoint, num_epochs, num_ite):
    if epoch_start_from_checkpoint == num_epochs - 1 and ite_start_from_checkpoint == num_ite - 1:
        raise ValueError("The loaded checkpoint refers to a completed training.")
    elif ite_start_from_checkpoint == num_ite - 1:
        ite_start_from_checkpoint = 0
        epoch_start_from_checkpoint += 1
    else:
        ite_start_from_checkpoint += 1
    return epoch_start_from_checkpoint, ite_start_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    config = parse_config.parse_config(args.config_path)

    deterministic = coalesce(config["deterministic"], False)
    torch.use_deterministic_algorithms(deterministic)

    trainloader, testloader, _ = config["data"]["dataset_loader"](**config["data"]["hyperparameters"])

    tot_training_ite_without_burnout = len(trainloader) * config["train"]["epochs"]

    if config["train"]["optim"]["scheduler"]["class"].__name__ in  ("CyclicLR", "CyclicLRWithBurnout"):
        cyc.cyclical_lr_determine_up_and_down_size(config, tot_training_ite_without_burnout)
        cyc.determine_base_lr(config)
        cyc.cyclical_lr_determine_total_steps(config, config["train"]["epochs"], len(trainloader))


    net_module = config["net_class"](NUM_CLASSES[config["data"]["dataset"]], **config["net_hyperparameters"])

    make_subdirectory(config["train"]["checkpoint_path"])
    make_subdirectory(config["train"]["final_model_save_path"])

    if config["train"]["pruning"]["scheduler"]["hyperparameters"].get("tot_num_pruning_ite") is None:
        config["train"]["pruning"]["scheduler"]["hyperparameters"]["tot_num_pruning_ite"] = tot_training_ite_without_burnout // config["train"]["pruning"]["hyperparameters"]["pruning_frequency"]
    config["train"]["pruning"]["scheduler"]["pruning_frequency"] = config["train"]["pruning"]["hyperparameters"]["pruning_frequency"]

    print(config["train"]["pruning"]["scheduler"]["hyperparameters"].get("tot_num_pruning_ite"), config["train"]["pruning"]["scheduler"]["pruning_frequency"])

    net = Model(
        module=net_module,
        optimizer_class=config["train"]["optim"]["class"],
        optimizer_kwargs=config["train"]["optim"]["hyperparameters"],
        lr_scheduler_class=config["train"]["optim"]["scheduler"]["class"],
        lr_scheduler_kwargs=config["train"]["optim"]["scheduler"]["hyperparameters"],
        lr_scheduler_update_time=config["train"]["optim"]["scheduler"]["update_time"],
        loss_fn=config["train"]["loss"](),
        mask=None,
        mask_class=config["train"]["pruning"]["mask_class"],
        mask_kwargs={
            **config["train"]["pruning"]["hyperparameters"],
            "pruning_rate_schedule": config["train"]["pruning"]["scheduler"]["class"],
            "scheduling_kwargs": config["train"]["pruning"]["scheduler"]["hyperparameters"]
        },
        name=config["net"]

    )


    epoch_start = ite_start = 0
    if config["train"]["resume_training"]:
        if os.path.isfile(check_path:=config["train"]["checkpoint_path"]):
            epoch_start, ite_start = net.load_checkpoint(torch.load(check_path))
            epoch_start, ite_start = determine_epoch_and_ite_start(epoch_start, ite_start, config["train"]["epochs"] + config["train"]["burnout_epochs"], len(trainloader))


    net.train_model(
        trainloader=trainloader,
        num_epochs=config["train"]["epochs"],
        burnout_epochs=config["train"]["burnout_epochs"],
        telegram_config_path=config["util"]["telegram_config_path"],
        telegram_config_name=config["util"]["telegram_config_name"],
        checkpoint_path=config["train"]["checkpoint_path"],
        checkpoint_save_time=config["train"]["checkpoint_save_time"],
        final_model_path=config["train"]["final_model_save_path"],
        epoch_start=epoch_start,
        ite_start=ite_start,
        amp_args=config["train"]["amp_hyperparameters"]
    )

    net.evaluate(
        testloader=testloader,
        eval_loss=True,
        telegram_config_path=config["util"]["telegram_config_path"],
        telegram_config_name=config["util"]["telegram_config_name"],
    )

if __name__ == "__main__":
    main()
