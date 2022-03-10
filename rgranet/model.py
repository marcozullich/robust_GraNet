from numpy import gradient
import torch
from enum import Enum
from typing import Union
import os
from apex import amp
from collections import OrderedDict as Odict

from .pruning_mask import _Mask, GraNetMask, GradualPruningMask, LMMask, NoMask, RGraNetMask
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
        self.gradients_accumulator = None
        

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
    
    def _accumulate_grad(self):
        if self.gradients_accumulator is None:
            self.gradients_accumulator = Odict({n: p.grad.detach().clone() for n, p in self.filtered_named_parameters(self.mask.effective_params_to_prune)})
        else:
            for n, p in self.filtered_named_parameters(self.mask.effective_params_to_prune):
                self.gradients_accumulator[n] += p.grad.detach().clone()

    def _accumulate_grad_if_required(self):
        if hasattr(self.mask.scheduling, "is_waiting_for_regrowth") and self.mask.scheduling.is_waiting_for_regrowth() and hasattr(self.mask, "accumulate_gradients_before_regrowth") and self.mask.accumulate_gradients_before_regrowth:
                    self._accumulate_grad()
        
    def _drop_accumulated_grad(self):
        self.gradients_accumulator = None

    def _clip_grad_norm(self, norm:float):
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)

    
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
            
            data, labels = data.to(device), labels.to(device)
            
            update_mask_this_ite = (i + 1) % num_ite_optimizer_step == 0 and not burnout

            if self.mask is not None and (not isinstance(self.mask, NoMask)) and update_mask_this_ite:
                self.mask.step()
                if isinstance(self.mask, (RGraNetMask, GradualPruningMask)):
                    # RGraNet: prune before weights update and gradient computation
                    self.mask.prune()

            preds = self.net(data)

            loss = self.loss_fn(preds, labels)

            step_optimizer_this_ite = ((i + 1) % num_ite_optimizer_step == 0) and (not burnout)
            if step_optimizer_this_ite:
                self.optimizer.zero_grad()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if clip_grad_norm:
                self._clip_grad_norm(1)
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            
            self._accumulate_grad_if_required()

            if self.mask is not None and isinstance(self.mask, (RGraNetMask)) and update_mask_this_ite:
                # RGraNet: regrow after gradient computation, before weights update
                self.mask.regrow(named_gradients=self.gradients_accumulator)


            if step_optimizer_this_ite:
                if not isinstance(self.mask, NoMask):
                    self.mask.suppress_grad()
                self.optimizer.step()
                if not isinstance(self.mask, NoMask):
                    self.mask.apply()
            
            if isinstance(self.mask, GraNetMask):
                # GraNet: prune and regrow AFTER weights update
                self.mask.prune()
                self.mask.regrow()
            
            accuracy_meter.update(acc_fn(preds, labels), n=data.shape[0])
            loss_meter.update(loss.item(), n=data.shape[0])
            if self.scheduler_update_time == TrainingMilestone.END_ITERATION and step_optimizer_this_ite:
                self.scheduler.step()
            
            

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
            "device": next(iter(self.net.parameters())).device,
            "amp": amp.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        device = state_dict["device"]
        self.net.load_state_dict(state_dict["net"])
        self.net.to(device)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.logger.load_state_dict(state_dict["logger"])
        amp.load_state_dict(state_dict["amp"])
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
