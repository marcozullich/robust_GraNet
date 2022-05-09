import os
from collections import OrderedDict as Odict
from enum import Enum
from types import SimpleNamespace
from typing import Union

import torch
# from apex import amp

from .model_logger import DistributedLogger, Milestone
from .pruning_mask import GradientsAccumulationMethod, GradualPruningMask, GraNetMask, LMMask, NoMask, RGraNetMask, _Mask
from .pytorch_utils import accuracy as acc_fn
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
        # name:str="",
        distributed_device:Union[str, torch.device]=None,
        is_main_device:bool=True,
    ):
        super().__init__()
        self.net = module
        optimizer_kwargs = coalesce(vars(optimizer_kwargs), {})
        self.optimizer = optimizer_class(self.net.parameters(), **optimizer_kwargs)
        lr_scheduler_kwargs = coalesce(vars(lr_scheduler_kwargs), {})
        self.scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
        self.scheduler_update_time = lr_scheduler_update_time
        self.loss_fn = loss_fn

        mask_kwargs = coalesce(mask_kwargs, {})

        self.parameters_names = self._get_parameters_names()

        self.mask = mask
        if self.mask is None:
            self.mask = mask_class(**mask_kwargs, net=self)
        
        # self.name = name
        self.gradients_accumulator = None

        self.scaler = torch.cuda.amp.GradScaler()

        self.is_main_device = is_main_device
        self.distributed_device = distributed_device
        # If distributed training, the device is set at initialization and cannot be changed in-training
        if distributed_device is not None:
            self.net.cuda(distributed_device)
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[distributed_device])
        elif torch.cuda.device_count() > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))

    def _decouple_distributed_structures(self):
        if isinstance(self.net, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            return self.net.module
        else:
            return self.net

    def _get_parameters_names(self):
        net = self._decouple_distributed_structures()
        return [n for n, _ in net.named_parameters()]
    
    def filtered_named_parameters(self, names):
        '''
        Returns a generator of tuples of (name, parameter) where the parameters are filtered by the specified names.
        '''
        net = self._decouple_distributed_structures()
        for n, p in net.named_parameters():
            if any([na in n for na in names]):
                yield n, p

    def forward(self, data):
        return self.net(data)
    
    def numel(self):
        return sum([p.numel() for p in self.net.parameters()])

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
        if hasattr(self.mask, "gradients_accumulation_method") and self.mask.gradients_accumulation_method != GradientsAccumulationMethod.NEVER:
            if self.mask.gradients_accumulation_method == GradientsAccumulationMethod.ALWAYS:
                self._accumulate_grad()
            elif self.mask.gradients_accumulation_method == GradientsAccumulationMethod.BETWEEN_PRUNE_AND_REGROWTH:
                if hasattr(self.mask.scheduling, "is_waiting_for_regrowth") and self.mask.scheduling.is_waiting_for_regrowth():
                    self._accumulate_grad()
            else:
                raise ValueError(f"Unknown gradients accumulation method {self.mask.gradients_accumulation_method}")
        
    def _drop_accumulated_grad(self):
        self.gradients_accumulator = None

    def _drop_accumulated_grad_if_required(self):
        if self.mask.need_gradient_reset:
            self._drop_accumulated_grad()
            self.mask.need_gradient_reset = False

    def _clip_grad_norm(self, norm:float):
        '''
        Operates clipping on the gradients w.r.t. the speicifed norm (L2).
        '''
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), norm)

    def _get_device(self):
        '''
        Used to inspect the device of the model in case of distributed training.
        '''
        if self.distributed_device is None:
            device = next(iter(self.parameters())).device
        else:
            device = self.distributed_device
        return device

    def _to_device(self, device):
        '''
        Sends the model to the desired device if not training in a distributed fashion; otherwise, does nothing.
        '''
        if self.distributed_device is None and (not isinstance(self, torch.nn.parallel.DataParallel)):
            device = coalesce(device, use_gpu_if_available())
            self.net.to(device)
    
    def train_model(
        self,
        trainloader:torch.utils.data.DataLoader,
        num_epochs:int,
        device:Union[torch.device,str]=None,
        clip_grad_norm_before_epoch:int=None,
        checkpoint_path:str=None,
        checkpoint_save_time:TrainingMilestone=TrainingMilestone.END_EPOCH,
        final_model_path:str=None,
        epoch_start:int=0,
        ite_start:int=0,
        burnout_epochs:int=0,
        half_precision:bool=False,
        distributed_debug_mode:bool=False,
        distributed_debug_mode_config:SimpleNamespace=None,
        ite_print:int=None
    ):
        if distributed_debug_mode:
            assert distributed_debug_mode_config is not None, "distributed_debug_mode_config must be provided if distributed_debug_mode is True"
        if not half_precision:
            self.scaler._enabled = False
        
        print("Starting training...")
        

        if clip_grad_norm_before_epoch is None:
            clip_grad_norm_before_epoch = num_epochs + burnout_epochs + 1
        
        self._to_device(device)

        self.net.train()
        
        checkpoint_save_path_iteration = checkpoint_path if checkpoint_save_time == TrainingMilestone.END_ITERATION else None

        for epoch in range(epoch_start, tot_epochs := num_epochs + burnout_epochs):
            if hasattr(trainloader, "sampler") and hasattr(trainloader.sampler, "set_epoch"):
                trainloader.sampler.set_epoch(epoch)
            self.train_epoch(
                epoch=epoch,
                trainloader=trainloader,
                clip_grad_norm_before_epoch=clip_grad_norm_before_epoch,
                checkpoint_path=checkpoint_save_path_iteration,
                ite_start=ite_start,
                burnout=epoch >= num_epochs,
                epochs=tot_epochs,
                distributed_debug_mode=distributed_debug_mode,
                distributed_debug_mode_config=distributed_debug_mode_config,
                ite_print=ite_print,
            )

            if self.scheduler_update_time == TrainingMilestone.END_EPOCH:
                self.scheduler.step()
            
            if self.is_main_device and checkpoint_path is not None and checkpoint_save_time == TrainingMilestone.END_EPOCH:
                torch.save(self.state_dict(), checkpoint_path)

        if final_model_path is not None and self.is_main_device:
            torch.save(self.state_dict(), final_model_path)
            if checkpoint_path is not None:
                os.remove(checkpoint_path)
    
    def train_epoch(
        self,
        epoch:int,
        trainloader:torch.utils.data.DataLoader,
        clip_grad_norm_before_epoch:int=0,
        checkpoint_path:str=None,
        ite_start:int=0,
        burnout:bool=False,
        ite_print:int=None,
        epochs:int=None,
        distributed_debug_mode:bool=False,
        distributed_debug_mode_config:SimpleNamespace=None,
    ):  
        # print(f"Epoch {epoch+1} started")
        logger = DistributedLogger()
        device = self._get_device()

        if ite_print is None:
            ite_print = len(trainloader)
        logger_header = f"{epoch+1}/{epochs}"

        # print("Fetching data")
        for i, (data, labels) in enumerate(logger.looper(trainloader, print_freq=ite_print, header=logger_header)):
            # print("Data fetched")
            if i < ite_start:
                continue
            
            # print("Zeroed optimizer")
            self.optimizer.zero_grad()

            # if i==0:
            #     print("Shapes", data.shape, labels.shape)

            # print("Moving data to device")
            data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            # print("Moved data to device")
            
            update_mask_this_ite = not burnout

            if self.mask is not None and (not isinstance(self.mask, NoMask)) and update_mask_this_ite:
                self.mask.step()
                if isinstance(self.mask, (RGraNetMask, GradualPruningMask)):
                    # RGraNet: prune before weights update and gradient computation
                    self.mask.prune()
                if distributed_debug_mode:
                    torch.save(self.mask.state_dict(), f"mask_state_dict_{distributed_debug_mode_config.jobno}_{distributed_debug_mode_config.rank}_{i}.pt")

            with torch.cuda.amp.autocast(True):
                # print("Forward pass")
                preds = self.net(data)
                # print("Loss calculation")
                loss = self.loss_fn(preds, labels)

            self.scaler.scale(loss).backward()

            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()

            torch.cuda.synchronize()

            if epoch < clip_grad_norm_before_epoch:
                self._clip_grad_norm(1)
            
            self._accumulate_grad_if_required()

            if self.mask is not None and isinstance(self.mask, (RGraNetMask)) and update_mask_this_ite:
                # RGraNet: regrow after gradient computation, before weights update
                if distributed_debug_mode:
                    torch.save(Odict({n: p.grad.detach().clone() for n, p in self.filtered_named_parameters(self.mask.effective_params_to_prune)}), f"grads_{distributed_debug_mode_config.jobno}_{distributed_debug_mode_config.rank}_{i}.pt")

                self.mask.regrow(named_gradients=self.gradients_accumulator)
                self._drop_accumulated_grad_if_required()


            # if step_optimizer_this_ite:
            if not isinstance(self.mask, NoMask):
                self.mask.suppress_grad()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if not isinstance(self.mask, NoMask):
                self.mask.apply()
            
            if isinstance(self.mask, GraNetMask):
                # GraNet: prune and regrow AFTER weights update
                self.mask.prune()
                self.mask.regrow()
            
            logger.update(loss=loss.item())
            logger.update(accuracy=acc_fn(preds, labels))

            if self.scheduler_update_time == TrainingMilestone.END_ITERATION: # and step_optimizer_this_ite:
                self.scheduler.step()

            if checkpoint_path is not None and self.is_main_device:
                torch.save(self.state_dict(epoch, i), checkpoint_path)

        logger.synchronize_between_processes()
        print("Averaged stats:", logger)

        return logger.to_dict()
    

    def evaluate(self, testloader:torch.utils.data.DataLoader,  eval_loss=True, adversarial_attack=None, device=None):
        self.eval()

        device = self._get_device()
        self._to_device(device)

        with torch.set_grad_enabled(adversarial_attack is not None):
            logger = DistributedLogger()
            for i, (data, labels) in enumerate(logger.looper(testloader, header="TEST", print_freq=len(testloader))):
                
                if i==0:
                    print("EVAL Shapes", data.shape, labels.shape)

                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(True):
                    preds = self.net(data)

                torch.cuda.synchronize()

                if eval_loss:
                    logger.update(loss=self.loss_fn(preds, labels).item())
                logger.update(accuracy=acc_fn(preds, labels))

        logger.synchronize_between_processes()
        print("Averaged stats:", logger)

        return logger.to_dict()

    def state_dict(self, epoch=None, ite=None):
        state_dict = {
            "net": self._decouple_distributed_structures().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "mask": self.mask.state_dict() if self.mask is not None else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "ite": ite
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):

        self._decouple_distributed_structures().load_state_dict(state_dict["net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.scaler.load_state_dict(state_dict["scaler"])
        if (mask_state_dict:=state_dict["mask"]) is not None:
            self.mask.load_state_dict(mask_state_dict)
    
    def load_trained_model(self, state_dict: dict):
        self._decouple_distributed_structures().load_state_dict(state_dict["net"])
        if (mask_state_dict:=state_dict["mask"]) is not None:
            self.mask.load_mask(state_dict["mask"], apply=True)


