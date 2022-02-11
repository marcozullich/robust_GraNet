from enum import Enum
from typing import Collection
from urllib import request
from functools import total_ordering
from tqdm import trange
from tqdm.contrib.telegram import trange as trange_telegram
import requests
import os



from .utils import yaml_load, DictOfLists

@total_ordering
class Verbosity(Enum):
    SILENT = 0
    VERBOSE_PRINT = 1
    VERBOSE_TELEGRAM = 2
    VERBOSE_ALL = 3

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return self.value < other.value
        if isinstance(other, int):
            return self.value < other
        raise NotImplementedError(f"Comparison between a Verbosity instance and an object of type {type(other)} is not implemented")

class Milestone(Enum):
    TRAIN_EPOCH = 0
    VALIDATION = 1
    VALIDATION_ADV = 2
    TEST = 3
    TEST_ADV = 4

MILESTONE_PRINT = {
    Milestone.TRAIN_EPOCH: "Train",
    Milestone.VALIDATION: "Validation",
    Milestone.VALIDATION_ADV: "Adv Validation",
    Milestone.TEST: "Test",
    Milestone.TEST_ADV: "Adv Test"
}



class ModelLogger():
    def __init__(self, logs_categories:Collection=["train_accuracy", "train_loss", "train_lr", "test_accuracy"],verbosity:Verbosity=Verbosity.VERBOSE_ALL):
        self.use_telegram = (verbosity >= Verbosity.VERBOSE_TELEGRAM)
        self.use_print = (verbosity == Verbosity.VERBOSE_ALL or verbosity >= Verbosity.VERBOSE_PRINT)
        self.logs = DictOfLists(logs_categories)
        self.token = None
        self.chatid = None

    def telegram_configure(self, config_path:str, config_name:str):
        config = yaml_load(config_path)[config_name]
        self.token = config["token"]
        self.chatid = config["chat_id"]
        self.use_telegram = True
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


    def _telegram_send(self, message):
        if not (hasattr(self, "token") and hasattr(self, "chatid")):
            raise ValueError("Could not send message. Use ModelLogger.telegram_configure(token_path, chatid_path) to configure the sender")
        # if not message.startswith("```"):
        #     message = "```" + message
        # if not message.endswith("```"):
        #     message += "```"


        send_url = f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={self.chatid}&text={message}"
        requests.get(send_url, verify=False)

    def _get_message_logs_end_epoch(self, milestone:Milestone):
        message = MILESTONE_PRINT[milestone] + " | "
        if milestone == Milestone.TRAIN_EPOCH:
            epoch = self.logs.len_key("train_accuracy")
            acc = self.logs.get_item("train_accuracy")
            loss = self.logs.get_item("train_loss")
            lr = self.logs.get_item("train_lr")
            message += f"Epoch {epoch} | Acc: {acc:.4f} | Loss: {loss:.4f}"

            if self.logs.get_item("mask sparsity") is not None:
                mask_sparsity = self.logs.get_item("mask sparsity")
                model_sparsity = self.logs.get_item("model sparsity")
                message += f" | Sparsity - mask: {mask_sparsity:.4f} - model: {model_sparsity:.4f}"
            if lr is not None:
                message += f" | LR: {lr[0]:.4f}"
        else:
            acc = self.logs.get_item("test_accuracy")
            loss = self.logs.get_item("test_loss")
            message += f"Acc: {acc:.4f}"
            if loss is not None:
                message += f" | Loss: {loss:.4f}"
        return message
    
    def log(self, dict_logs:dict, milestone:Milestone):
        for k, v in dict_logs.items():
            self.logs[k].append(v)
        if self.use_print or self.use_telegram:
            message = self._get_message_logs_end_epoch(milestone)
            if self.use_telegram:
                self._telegram_send(message)
            if self.use_print:
                print(message)
    
    def prompt_start(self, nepochs, net=None, save_path=None, epoch_start=0, ite_start=0, dtype="float32"):
        message = f"Starting training for {nepochs} epochs..."
        if net is not None:
            message += f"\n{net.name}\n"
            message += f"\tOptimizer: {net.optimizer.__class__.__name__} - lr: {net.optimizer.param_groups[0]['lr']}\n"
            message += f"\tScheduler: {net.scheduler.__class__.__name__}\n"
            # message += f"\tPrecision: {dtype}\n"
            message += f"\tMask: {net.mask.__class__.__name__}\n"
            if hasattr(net.mask, "scheduling"):
                message += f"\tMask Scheduler: {net.mask.scheduling.__class__.__name__}\n"
                message += f"\tTarget sparsity: {net.mask.scheduling.final_sparsity}\n"
            if save_path is not None:
                message += f"\tSave path: {os.path.expanduser(os.path.expandvars(save_path))}\n"
            if epoch_start >  0 or ite_start > 0:
                message += f"**Resuming from epoch {epoch_start} at iteration {ite_start}**\n"
            if hasattr(net.mask, "effective_params_to_prune"):
                message += f"\tEffective params to prune:\n{net.mask.effective_params_to_prune}\n"

        if self.use_print:
            print(message)
        if self.use_telegram:
            self._telegram_send(message)
        
    
    def add_log_categories(self, *categories):
        self.logs.add_keys(*categories)

    def state_dict(self):
        state_dict = {
            "logs": self.logs,
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.logs = state_dict["logs"]

