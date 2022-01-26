import torch
from torch import nn
import argparse
import os

from rgranet import data
from rgranet.model import Model, LRSchedulerUpdateTime
from rgranet.pruning_rate_schedule import PruningRateCubicSchedulingWithRegrowth as SchedulingRegrowth

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
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing the MNIST dataset. If not given, it will default to $DATASET_PATH/mnist.")
    parser.add_argument("--batch_train", type=int, default=128)
    parser.add_argument("--batch_test", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ite_pruning_frequency", type=int, default=1)
    parser.add_argument("--use_telegram", action="store_true", 
    default=False)
    parser.add_argument("--telegram_tokens_folder", default=None, type=str, help="Folder containing telegram tokens (chatid.txt and token.txt files)")
    args = parser.parse_args()

    if args.data_dir is None:
        if os.environ.get("DATASET_PATH") is None:
            raise ValueError("Please set the environment variable DATASET_PATH to the dir containing the `mnist` folder.")
        data_dir = os.path.join(os.environ["DATASET_PATH"], "mnist")
    else:
        data_dir = args.data_dir
    
    
    telegram_token_path, telegram_chatid_path = set_telegram_tokens(args.use_telegram, args.telegram_tokens_folder)
    

    trainloader, testloader, _ = data.MNIST_DataLoaders(data_dir, batch_size_train=args.batch_train, batch_size_test=args.batch_test)

    tot_pruning_ite = args.epochs * len(trainloader) // args.ite_pruning_frequency

    net_module = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 64),
        nn.Linear(64, 10)
    )
    net = Model(
        module=net_module,
        optimizer_class=torch.optim.RAdam,
        lr_scheduler_class=torch.optim.lr_scheduler.CyclicLR,
        lr_scheduler_update_time=LRSchedulerUpdateTime.END_ITERATION,
        loss_fn=nn.CrossEntropyLoss(),
        mask=None,
        mask_class = {
            "init_pruning_rate": 0.0,
            "params_to_prune": [".weight"],
            "pruning_rate_schedule": SchedulingRegrowth,
            "scheduling_kwargs": {
                "initial_sparsity": 0.0,
                "final_sparsity": 0.3,
                "tot_num_pruning_ite": tot_pruning_ite,
                "pruning_frequency": args.ite_pruning_frequency
            },
            "n_steps_prune": args.ite_pruning_frequency
        }
    )

    net.train_model(
        trainloader=trainloader,
        num_epochs=args.num_epochs,
        telegram_chat_token_path=telegram_token_path,
        telegram_chatid_path=telegram_chatid_path
    )
    
