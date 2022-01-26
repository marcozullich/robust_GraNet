from enum import Enum
from typing import Collection
import telegram_send as ts
from functools import total_ordering
from tqdm import trange
from tqdm.contrib.telegram import trange as trange_telegram
import requests

from .utils import get_file_content, DictOfLists

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
        self.verbosity = verbosity
        self.use_telegram = (verbosity >= Verbosity.VERBOSE_TELEGRAM)
        self.use_print = (verbosity == Verbosity.VERBOSE_ALL or verbosity >= Verbosity.VERBOSE_PRINT)
        self.logs = DictOfLists(logs_categories)

    def telegram_configure(self, token_path, chatid_path):
        self.token = get_file_content(token_path, f"Could not configure telegram logger. Token file not found at {token_path}")
        self.chatid = get_file_content(chatid_path, f"Could not configure telegram logger. Chat ID file not found at {chatid_path}")
        self.use_telegram = True
    


    def _telegram_send(self, message):
        if not (hasattr(self, "token") and hasattr(self, "chatid")):
            raise ValueError("Could not send message. Use ModelLogger.telegram_configure(token_path, chatid_path) to configure the sender")
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
            if lr is not None:
                message += f" | LR: {lr:.4f}"
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
    
    def log_range_progress(self, *args):
        if self.use_telegram:
            try:
                trange_telegram(*args, token=self.token, chat_id=self.chatid)
            except AttributeError:
                raise ValueError("Could not send message. Use ModelLogger._telegram_configure(token_path, chatid_path) to configure the sender")
        if self.use_print:
            trange(*args)
    
    def add_log_categories(self, *categories):
        self.logs.add_keys(*categories)