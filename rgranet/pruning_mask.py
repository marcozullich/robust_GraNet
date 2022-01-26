from typing import Collection, Union
from collections import OrderedDict as Odict
import torch

from . import pruning_rate_schedule as schedule
from .utils import coalesce
from .neuroregeneration import gradient_based_neuroregeneration



class _Mask():
    def __init__(
        self,
        init_pruning_rate:float,
        net:torch.nn.Module,
        params_to_prune:Collection=None,
        pruning_rate_schedule=schedule.NoPruningRateAnnealing,
        scheduling_kwargs:dict=None,
        is_global:bool=True,
        n_steps_prune:int=1
    ):
        scheduling_kwargs = coalesce(scheduling_kwargs, {"initial_pruning_rate": init_pruning_rate})
        pruning_rate_schedule = coalesce(pruning_rate_schedule, schedule.NoPruningRateAnnealing)
        self.p = init_pruning_rate
        self.params_to_prune = params_to_prune
        self.scheduling = pruning_rate_schedule(**scheduling_kwargs)
        self.is_global = is_global
        self.steps = 0
        self.n_steps_prune = n_steps_prune

        self.params_to_prune = coalesce(self.params_to_prune, net.parameters_names)
        self.effective_params_to_prune = []
        self.net = net
        for name, _ in net.filtered_named_parameters(self.params_to_prune):
            self.effective_params_to_prune.append(name)
        self.mask = self._init_mask()

    def _init_mask(self):
        mask = Odict()
        for (name, param) in self.net.filtered_named_parameters(self.effective_params_to_prune):
            mask[name] = torch.ones_like(param).bool()
        return mask

    def apply(self):
        for (_, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask.items()):
            param.data *= msk

    def suppress_grad(self):
        for (_, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask):
            param.grad *= msk

    def _criterion(self, *params) -> Odict:
        raise NotImplementedError("This is an abstract class")

    def _update(self):
        if self.is_global:
            parameters = self.net.filtered_named_parameters(self.effective_params_to_prune)
            self.mask = self._criterion(*parameters)
        else:
            new_mask = Odict()
            for (name, param), in self.net.filtered_named_parameters(self.effective_params_to_prune):
                new_mask[name] = self._criterion({name: param})
            self.mask = new_mask

    def step(self):
        if self.steps % self.n_steps_prune == 0:
            self._update()
            self.apply()
            if self.scheduling.regrowth_rate is not None and self.scheduling.regrowth_rate > 0.0:
                gradient_based_neuroregeneration(self.net, self.params_to_prune, self.scheduling.regrowth_rate, self.is_global)
            self.p = self.scheduling.step()
        self.steps += 1
    
    def get_nonzero_weights_count(self):
        return sum(torch.nonzero(msk).size(0) for (_, msk) in self.mask.items())
    
    def get_zero_weights_count(self):
        return self.numel() - self.get_nonzero_weights_count()

    def numel(self):
        return sum([msk.numel() for (_, msk) in self.mask.items()])

    def get_mask_sparsity(self):
        return self.get_zero_weights_count() / self.numel()
    
    def get_model_sparsity(self):
        return self.get_zero_weights_count() / self.net.numel()

    def names(self):
        return self.mask.keys()
    
    def regenerate(self, regeneration_mask:Odict):
        for (name, regen_msk) in regeneration_mask.items():
            self.mask[name] = self.mask[name].logical_or(regen_msk)
        self.apply()

    def __getitem__(self, key):
        return self.mask[key]
        
class LMMask(_Mask):

    def _criterion(self, *params) -> Odict:
        flattened_abs_params = torch.cat([param[self.mask[name]].abs() for name, param in params])
        quantile = flattened_abs_params.kthvalue(int(self.p * flattened_abs_params.numel()))[0]
        return Odict({name: param.abs() > quantile for name, param in params})
