import torch
from torch import nn
import argparse
import os

from rgranet import data
from rgranet.model import Model, TrainingMilestone
from rgranet import pruning_mask as msk
from rgranet.pruning_rate_schedule import PruningRateCubicSchedulingWithRegrowth as SchedulingRegrowth
from rgranet.utils import coalesce

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Directory containing the MNIST dataset. If not given, it will default to $DATASET_PATH/mnist.")
    parser.add_argument("--batch_train", type=int, default=128)
    parser.add_argument("--batch_test", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ite_pruning_frequency", type=int, default=1)
    parser.add_argument("--ite_regrow_frequency", type=int, default=None, help="Iterations interval for regrowth. If None, it defaults to --ite_pruning_frequency.")
    parser.add_argument("--use_telegram", action="store_true", 
    default=False)
    parser.add_argument("--telegram_tokens_folder", default=None, type=str, help="Folder containing telegram tokens (chatid.txt and token.txt files)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument("--cyclic_lr_min_lr", type=float, default=0.0, help="Minimum learning rate for the cyclic learning rate scheduler")
    parser.add_argument("--cyclic_lr_max_lr", type=float, default=0.2, help="Maximum learning rate for the cyclic learning rate scheduler")
    parser.add_argument("--cyclic_lr_up_ratio", type=float, default=1.0, help="Ratio of the upside phase of the cyclic LR schedule w.r.t. downward phase.")
    parser.add_argument("--target_sparsity", type=float, default=0.9, help="Target sparsity for the pruning rate schedule")
    
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.cyclic_lr_max_lr = 0.1
        args.use_telegram = True
        args.telegram_tokens_folder = "./telegram_config"
        args.ite_pruning_frequency = 10

    if args.root is None:
        if os.environ.get("DATASET_PATH") is None:
            raise ValueError("Please set the environment variable DATASET_PATH to the dir containing the `mnist` folder.")
        root = os.path.join(os.environ["DATASET_PATH"], "mnist")
    else:
        root = args.root
    
    args.ite_regrow_frequency = coalesce(args.ite_regrow_frequency, args.ite_pruning_frequency)

    telegram_token_path, telegram_chatid_path = set_telegram_tokens(args.use_telegram, args.telegram_tokens_folder)
    

    trainloader, testloader, _ = data.MNIST_DataLoaders(root, batch_size_train=args.batch_train, batch_size_test=args.batch_test)

    tot_training_ite = len(trainloader) * args.epochs
    tot_pruning_ite = tot_training_ite // args.ite_pruning_frequency

    net_module = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    cyclic_lr_up_size = int(args.cyclic_lr_up_ratio * tot_training_ite // (args.cyclic_lr_up_ratio + 1))
    cyclic_lr_down_size = tot_training_ite - cyclic_lr_up_size

    net = Model(
        module=net_module,
        optimizer_class=torch.optim.SGD,
        optimizer_kwargs={
            "lr": args.cyclic_lr_max_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay
        },
        # optimizer_class=torch.optim.Adam,
        # optimizer_kwargs={},
        lr_scheduler_class=torch.optim.lr_scheduler.CyclicLR,
        lr_scheduler_kwargs={
            "base_lr": args.cyclic_lr_min_lr,
            "max_lr": args.cyclic_lr_max_lr,
            "step_size_up": cyclic_lr_up_size,
            "step_size_down": cyclic_lr_down_size,
        },
        lr_scheduler_update_time=TrainingMilestone.END_ITERATION,
        # lr_scheduler_class=torch.optim.lr_scheduler.MultiStepLR,
        # lr_scheduler_kwargs={
        #     "milestones": [6, 8]
        # },
        # lr_scheduler_update_time=TrainingMilestone.END_EPOCH,
        loss_fn=nn.CrossEntropyLoss(),
        mask=None,
        mask_class=msk.LMMask,
        mask_kwargs = {
            "init_pruning_rate": 0.0,
            "params_to_prune": [".weight"],
            "pruning_rate_schedule": SchedulingRegrowth,
            "scheduling_kwargs": {
                "initial_sparsity": 0.0,
                "final_sparsity": args.target_sparsity,
                "tot_num_pruning_ite": tot_pruning_ite,
                "pruning_frequency": args.ite_pruning_frequency
            },
            "n_steps_each_prune": args.ite_pruning_frequency,
            "n_steps_each_regrow": args.ite_pruning_frequency,
        },
        name="FCN4ReLU"
    )


    net.train_model(
        trainloader=trainloader,
        num_epochs=args.epochs,
        telegram_chat_token_path=telegram_token_path,
        telegram_chatid_path=telegram_chatid_path
    )
    
    net.evaluate(
        testloader=testloader,
        eval_loss=True,
        telegram_chat_token_path=telegram_token_path,
        telegram_chatid_path=telegram_chatid_path
    )
