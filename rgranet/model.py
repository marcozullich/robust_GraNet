from mimetypes import init
import torch
from enum import Enum

from .pruning_mask import _Mask, LMMask
from .pytorch_utils import AverageMeter, correct_preds
from .model_logger import ModelLogger, Verbosity, Milestone
from .pruning_rate_schedule import _PruningRateScheduling, PruningRateCosineScheduling
from .utils import coalesce

class LRSchedulerUpdateTime(Enum):
    END_EPOCH = 0
    END_ITERATION = 1



class Model(torch.nn.Module):
    def __init__(
        self,
        module:torch.nn.Module,
        optimizer_class=None,
        optimizer_kwargs=None,
        lr_scheduler_class=None,
        lr_scheduler_kwargs=None,
        lr_scheduler_update_time:LRSchedulerUpdateTime=LRSchedulerUpdateTime.END_EPOCH,
        loss_fn:torch.nn.Module=None,
        mask:_Mask=None,
        mask_class=LMMask,
        mask_kwargs:dict=None,
        verbose=True
    ):
        super().__init__()
        self.net = module
        optimizer_kwargs = coalesce(optimizer_kwargs, {})
        self.optimizer = optimizer_class(self.net.parameters(), **optimizer_kwargs)
        lr_scheduler_kwargs = coalesce(lr_scheduler_kwargs, {})
        self.scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
        self.scheduler_update_time = lr_scheduler_update_time
        self.loss_fn = loss_fn

        mask_kwargs = coalesce(mask_kwargs, {})
        

        self.parameters_names = self._get_parameters_names()
        self.logger = ModelLogger(logs_categories=[], verbosity=Verbosity.VERBOSE_PRINT if verbose else Verbosity.SILENT)

        self.mask = mask
        if self.mask is None:
            self.mask = mask_class(**mask_kwargs, net=self)
        

        

    def _get_parameters_names(self):
        return [n for n, _ in self.net.named_parameters()]
    
    def filtered_named_parameters(self, names):
        for n, p in self.net.named_parameters():
            if any([na in n for na in names]):
                yield n, p

    def forward(self, data):
        return self.net(data)
    
    def numel(self):
        return sum([p.numel() for p in self.net.parameters()])

    def _determine_logger_verbosity(verbose, telegram_chat_token_path):
        if verbose and telegram_chat_token_path is not None:
            return Verbosity.VERBOSE_ALL
        if verbose:
            return Verbosity.VERBOSE_PRINT
        if telegram_chat_token_path is not None:
            logger_verbosity = Verbosity.VERBOSE_TELEGRAM
        return Verbosity.SILENT



    def train_model(self, trainloader:torch.utils.data.DataLoader, num_epochs:int, num_ite_optimizer_step:int=1, telegram_chat_token_path=None, telegram_chatid_path=None,  **kwargs):
        self.train()

        if telegram_chat_token_path is not None and telegram_chatid_path is not None:
            self.logger.telegram_configure(telegram_chat_token_path)
        self.logger.add_log_categories(("train_accuracy", "train_loss", "train_lr"))

        for epoch in self.logger.log_range_progress(num_epochs):
            accuracy_meter = AverageMeter()
            loss_meter = AverageMeter()
            accuracy, loss_val = self.train_epoch(trainloader, epoch, accuracy_meter, loss_meter, num_ite_optimizer_step=num_ite_optimizer_step)

            self.logger.log({"train_accuracy": accuracy, "train_loss": loss_val, "train_lr": self.scheduler.get_lr()[0]}, milestone=Milestone.TRAIN_EPOCH)

            if self.scheduler_update_time == LRSchedulerUpdateTime.END_EPOCH:
                self.scheduler.step(epoch)
    
    def train_epoch(self, trainloader:torch.utils.data.DataLoader, accuracy_meter:AverageMeter, loss_meter:AverageMeter, num_ite_optimizer_step:int=1, **kwargs):
        for i, (data, labels) in enumerate(trainloader):
            preds = self.forward(data)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            if self.mask is not None:
                self.mask.suppress_grad()
            if (i + 1) % num_ite_optimizer_step == 0:
                self.optimizer.zero_grad()
                self.optimizer.step()
            accuracy_meter.update(correct_preds(preds, labels), n=len(labels))
            loss_meter.update(loss.item())
            if self.scheduler_update_time == LRSchedulerUpdateTime.END_ITERATION:
                self.scheduler.step(i)
            if self.mask is not None:
                self.mask.step()
        return accuracy_meter.avg, loss_meter.avg
    
    def _set_up_logger_for_evaluation(self, validation:bool, is_adv:bool, eval_loss:bool, telegram_chat_token_path, telegram_chatid_path):
        if telegram_chat_token_path is not None and telegram_chatid_path is not None:
            self.logger.telegram_configure(telegram_chat_token_path)
        if validation:
            if not is_adv:
                milestone = Milestone.VALIDATION
                category_prefix = "validation_"
            else:
                milestone = Milestone.VALIDATION_ADV
                category_prefix = "validation_adv_"
        else:
            if not is_adv:
                milestone = Milestone.TEST
                category_prefix = "test_"
            else:
                milestone = Milestone.TEST_ADV
                category_prefix = "test_adv_"
        
        self.logger.add_log_categories(category_prefix + "accuracy")
        if eval_loss:
            self.logger.add_log_categories(category_prefix + "loss")

        return milestone, category_prefix

    def evaluate(self, testloader:torch.utils.data.DataLoader, eval_loss=True, adversarial_attack=None, telegram_chat_token_path=None, telegram_chatid_path=None, validation=False, **kwargs):
        self.eval()

        milestone, category_prefix = self._set_up_logger_for_evaluation(validation, adversarial_attack is not None, eval_loss, telegram_chat_token_path, telegram_chatid_path)
        
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter() if eval_loss else None
        with torch.set_grad_enabled(adversarial_attack is not None):
            for data, labels in testloader:
                preds = self.forward(data)
                accuracy_meter.update(correct_preds(preds, labels), n=len(labels))
                if loss_meter is not None:
                    loss_meter.update(self.loss_fn(preds, labels).item())

        accuracy = accuracy_meter.avg
        loss_val = loss_meter.avg if eval_loss else None
        
        logging_dict = {category_prefix + "accuracy": accuracy}
        if eval_loss:
            logging_dict[category_prefix + "loss"] = loss_val
        self.logger.log(logging_dict, milestone=milestone)
        return accuracy, loss_val
