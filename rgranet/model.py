from mimetypes import init
from tabnanny import check
import torch
from enum import Enum
from typing import Union
import os
from apex import amp
import torchvision

from .pruning_mask import _Mask, LMMask, NoMask
from .pytorch_utils import AverageMeter, accuracy as acc_fn
from .model_logger import ModelLogger, Verbosity, Milestone
from .pytorch_utils import set_dtype_
from .utils import coalesce

class TrainingMilestone(Enum):
    START_TRAIN = 0
    START_EPOCH = 1
    START_ITERATION = 2
    END_EPOCH = 3
    END_ITERATION = 4
    END_TRAIN = 5

def use_gpu_if_available(gpu_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}")
    else:
        return torch.device("cpu")


class Model(torch.nn.Module):
    def __init__(
        self,
        module:torch.nn.Module,
        optimizer_class=None,
        optimizer_kwargs=None,
        lr_scheduler_class=None,
        lr_scheduler_kwargs=None,
        lr_scheduler_update_time:TrainingMilestone=TrainingMilestone.END_EPOCH,
        loss_fn:torch.nn.Module=None,
        mask:_Mask=None,
        mask_class=LMMask,
        mask_kwargs:dict=None,
        verbose:bool=True,
        name:str="",
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
        
        self.name = name
        

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

    def _clone_grad(self):
        return [n.grad.clone() for n in self.net.parameters()]
    
    def _overwrite_grad(self, grad):
        for p, g in zip(self.net.parameters(), grad):
            p.grad = g

    def train_model(
        self,
        trainloader:torch.utils.data.DataLoader,
        num_epochs:int,
        num_ite_optimizer_step:int=1,
        telegram_config_path=None,
        telegram_config_name=None,
        device:Union[torch.device,str]=None,
        clip_grad_norm:bool=True,
        checkpoint_path:str=None,
        checkpoint_save_time:TrainingMilestone=TrainingMilestone.END_EPOCH,
        final_model_path:str=None,
        epoch_start:int=0,
        ite_start:int=0,
        dtype:Union[str, torch.dtype]=torch.float32,
        amp_args=None,
        burnout_epochs:int=0,
        **kwargs
    ):
        device = coalesce(device, use_gpu_if_available())
        
        self.to(device)

        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, **amp_args)

        # set_dtype_(self, dtype)

        self.train()

        if telegram_config_path is not None and telegram_config_name is not None:
            self.logger.telegram_configure(telegram_config_path, telegram_config_name)
        self.logger.add_log_categories("epoch", "train_accuracy", "train_loss", "train_lr")

        if self.mask is not None and not isinstance(self.mask, NoMask):
            self.logger.add_log_categories("mask sparsity", "model sparsity")

        self.logger.prompt_start(
            nepochs=num_epochs + burnout_epochs,
            net=self,
            save_path=final_model_path,
            epoch_start=epoch_start,
            ite_start=ite_start,
            dtype=dtype
        )

        checkpoint_save_path_iteration = checkpoint_path if checkpoint_save_time == TrainingMilestone.END_ITERATION else None

        for epoch in range(epoch_start, num_epochs + burnout_epochs):
            accuracy_meter = AverageMeter()
            loss_meter = AverageMeter()
            accuracy, loss_val = self.train_epoch(
                trainloader=trainloader,
                epoch=epoch,
                accuracy_meter=accuracy_meter,
                loss_meter=loss_meter,
                device=device,
                num_ite_optimizer_step=num_ite_optimizer_step, 
                clip_grad_norm=clip_grad_norm, 
                checkpoint_path=checkpoint_save_path_iteration,
                ite_start=ite_start,
                burnout=epoch >= num_epochs
            )

            dict_log = {"train_accuracy": accuracy, "train_loss": loss_val, "train_lr": self.scheduler.get_last_lr()}

            if self.mask is not None and not isinstance(self.mask, NoMask):
                dict_log["mask sparsity"] = self.mask.get_mask_sparsity()
                dict_log["model sparsity"] = self.mask.get_model_sparsity()

            self.logger.log(dict_log, milestone=Milestone.TRAIN_EPOCH)

            if self.scheduler_update_time == TrainingMilestone.END_EPOCH:
                self.scheduler.step()
            
            if checkpoint_path is not None and checkpoint_save_time == TrainingMilestone.END_EPOCH:
                torch.save(self.checkpoint(epoch, len(trainloader)), checkpoint_path)

        if final_model_path is not None:
            torch.save(self.state_dict(), final_model_path)
            if checkpoint_path is not None:
                os.remove(checkpoint_path)
    
    def train_epoch(
        self,
        epoch:int,
        trainloader:torch.utils.data.DataLoader,
        accuracy_meter:AverageMeter,
        loss_meter:AverageMeter,
        device,
        num_ite_optimizer_step:int=1,
        clip_grad_norm:bool=True,
        checkpoint_path:str=None,
        ite_start:int=0,
        dtype:Union[str, torch.dtype]=torch.float32,
        burnout:bool=False,
        **kwargs
    ):
        # times = []
        for i, (data, labels) in enumerate(trainloader):
            if i < ite_start:
                continue
            
            # set_dtype_(data, dtype)
            # set_dtype_(labels, dtype)
            data, labels = data.to(device), labels.to(device)
            
            
            update_mask_this_ite = (i + 1) % num_ite_optimizer_step == 0 and not burnout

            if self.mask is not None and (not isinstance(self.mask, NoMask)) and update_mask_this_ite:
                self.mask.step()
                # print(i, self.mask.scheduling.current_pruning_rate, self.mask.scheduling.can_prune(), self.mask.scheduling.step_counter, self.mask.scheduling.current_sparsity, self.mask.scheduling.regrowth_rate)
                self.mask.prune()
                # if self.mask.scheduling.supports_regrowth:
                #     # grad_copy = self._clone_grad
                #     pass

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()
            preds = self.net(data)
            # end.record()
            # torch.cuda.synchronize()
            # times.append(start.elapsed_time(end))
            loss = self.loss_fn(preds, labels)

            step_optimizer_this_ite = ((i + 1) % num_ite_optimizer_step == 0) and (not burnout)
            if step_optimizer_this_ite:
                self.optimizer.zero_grad()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)

            if self.mask is not None and not isinstance(self.mask, NoMask) and update_mask_this_ite:
                self.mask.regrow()


            if step_optimizer_this_ite:
                if not isinstance(self.mask, NoMask):
                    self.mask.suppress_grad()
                self.optimizer.step()
                self.mask.apply()
            
            accuracy_meter.update(acc_fn(preds, labels), n=data.shape[0])
            loss_meter.update(loss.item(), n=data.shape[0])
            if self.scheduler_update_time == TrainingMilestone.END_ITERATION and step_optimizer_this_ite:
                self.scheduler.step()
            # if self.mask is not None and update_mask_this_ite:
                # if grad_copy is not None:
                #     self._overwrite_grad(grad_copy)
                # self.mask.step()
            
            

            if checkpoint_path is not None:
                torch.save(self.checkpoint(epoch, i), checkpoint_path)

        # print(f"Average time of inference {sum(times)/len(times)}")

        return accuracy_meter.avg, loss_meter.avg
    
    def _set_up_logger_for_evaluation(self, validation:bool, is_adv:bool, eval_loss:bool, telegram_config_path, telegram_config_name):
        if telegram_config_path is not None and telegram_config_name is not None:
            self.logger.telegram_configure(telegram_config_path, telegram_config_name)
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

    def evaluate(self, testloader:torch.utils.data.DataLoader,  eval_loss=True, adversarial_attack=None, telegram_config_path=None, telegram_config_name=None, validation=False, device=None, dtype:Union[str, torch.dtype]=torch.float32 ,**kwargs):
        self.eval()
        device = coalesce(device, use_gpu_if_available())

        # set_dtype_(self, dtype)
        self.to(device)

        milestone, category_prefix = self._set_up_logger_for_evaluation(validation, adversarial_attack is not None, eval_loss, telegram_config_path, telegram_config_name)
        
        accuracy_meter = AverageMeter()
        loss_meter = AverageMeter() if eval_loss else None
        with torch.set_grad_enabled(adversarial_attack is not None):
            for data, labels in testloader:
                # set_dtype_(data, dtype)
                # set_dtype_(labels, dtype)
                data = data.to(device)
                labels = labels.to(device)

                preds = self.forward(data)
                accuracy_meter.update(acc_fn(preds, labels), n=len(labels))
                if loss_meter is not None:
                    loss_meter.update(self.loss_fn(preds, labels).item(), n=len(labels))

        accuracy = accuracy_meter.avg
        loss_val = loss_meter.avg if eval_loss else None
        
        logging_dict = {category_prefix + "accuracy": accuracy}
        if eval_loss:
            logging_dict[category_prefix + "loss"] = loss_val
        self.logger.log(logging_dict, milestone=milestone)
        return accuracy, loss_val

    def state_dict(self):
        state_dict = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "logger": self.logger.state_dict(),
            "mask": self.mask.state_dict() if self.mask is not None else None,
            "device": next(iter(self.net.parameters())).device
        
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        device = state_dict["device"]
        self.net.load_state_dict(state_dict["net"])
        self.net.to(device)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.logger.load_state_dict(state_dict["logger"])
        if (mask_state_dict:=state_dict["mask"]) is not None:
            self.mask.load_state_dict(mask_state_dict)

    def checkpoint(self, epoch, ite):
        checkpoint_dict = {
            "epoch": epoch,
            "ite": ite,
            "state_dict": self.state_dict(),
        }
        return checkpoint_dict

    def load_checkpoint(self, checkpoint):
        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint["epoch"], checkpoint["ite"]
