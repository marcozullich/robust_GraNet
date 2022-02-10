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
        pruning_rate_schedule=schedule.NoPruningRateScheduling,
        scheduling_kwargs:dict=None,
        is_global:bool=True,
        pruning_frequency:int=1,
        regrowth_frequency:int=1,
        step_start_prune:int=0,
        step_start_regrow:int=0

    ):
        scheduling_kwargs = coalesce(scheduling_kwargs, {"initial_pruning_rate": init_pruning_rate})
        scheduling_kwargs["pruning_frequency"] = pruning_frequency

        pruning_rate_schedule = coalesce(pruning_rate_schedule, schedule.NoPruningRateScheduling)
        self.p = init_pruning_rate
        self.params_to_prune = params_to_prune
        self.scheduling = pruning_rate_schedule(**scheduling_kwargs)
        self.is_global = is_global
        self.pruning_frequency = pruning_frequency
        self.regrowth_frequency = regrowth_frequency
        # self.step_start_prune = step_start_prune
        # self.step_start_regrow = step_start_regrow

        self.params_to_prune = coalesce(self.params_to_prune, net.parameters_names)
        self.effective_params_to_prune = []
        self.net = net
        for name, _ in net.filtered_named_parameters(self.params_to_prune):
            self.effective_params_to_prune.append(name)
        self.mask = self._init_mask()
        self.device = next(iter(self.net.parameters())).device

    def _init_mask(self):
        mask = Odict()
        for (name, param) in self.net.filtered_named_parameters(self.effective_params_to_prune):
            mask[name] = torch.ones_like(param).bool()
        return mask

    def apply(self):
        for (_, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask.items()):
            msk = msk.to(param.device)
            param.data *= msk
            msk = msk.cpu()

    def suppress_grad(self):
        for (_, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask.items()):
            msk = msk.to(param.grad.device)
            param.grad *= msk
            msk = msk.cpu()


    def _criterion(self, *params) -> Odict:
        raise NotImplementedError("This is an abstract class")

    def _update(self):
        if self.is_global:
            parameters = self.net.filtered_named_parameters(self.effective_params_to_prune)
            new_mask = self._criterion(*parameters)
        else:
            new_mask = Odict()
            for (name, param), in self.net.filtered_named_parameters(self.effective_params_to_prune):
                new_mask[name] = self._criterion({name: param})
        # self.mask_delta = Odict({n: m.logical_xor(new_mask) for n, m in self.mask.items()})
        # prev_sparsity = self.get_mask_sparsity()
        self.mask = new_mask
        # curr_sparsity = self.get_mask_sparsity()
        # self.delta_sparsity = curr_sparsity - prev_sparsity

    def prune(self):
        if self.p > 0.0:
            self._update()
            self.apply()
            # print(f"\tsparsity after pruning {self.get_mask_sparsity():.6f}")

    def step(self):
        self.scheduling.step()
        self.p = self.scheduling.current_pruning_rate
    
    def regrow(self):
        if self.p > 0.0:
            regen_mask = gradient_based_neuroregeneration(self.net, self.params_to_prune, self.scheduling.regrowth_rate, self.is_global)
            self.regenerate(regen_mask)
            # print(f"\tsparsity after regrowth {self.get_mask_sparsity():.6f}")
        
 
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
            self.mask[name].logical_or_(regen_msk)
        self.apply()

    def __getitem__(self, key):
        return self.mask[key]
    
    def state_dict(self):
        return {
            "mask": self.mask,
            "scheduling": self.scheduling.state_dict(),
        }

    def load_state_dict(self, state_dict, ):
        self.mask = state_dict["mask"]
        self.scheduling.load_state_dict(state_dict["scheduling"])
        
class LMMask(_Mask):

    def _criterion(self, *params) -> Odict:
        flattened_abs_params = torch.cat([param[self.mask[name]].abs() for name, param in params])
        index = int(self.p * flattened_abs_params.numel())
        # if index is 0, the pruning rate is too small to prune anything
        pth_quantile = None
        if index > 0:
            # print(f"PRUNE: {self.p:.6f} ++ {self.scheduling.regrowth_rate:.6f} - num params {flattened_abs_params.numel()} - index {index} - spa {self.get_mask_sparsity():.6f}")
            pth_quantile = flattened_abs_params.kthvalue(index)[0]
            return Odict({name: param.abs() >= pth_quantile for name, param in params})
        # print(f"PRUNE: {self.p} - num params {flattened_abs_params.numel()} - index {index} ---")
        return self.mask


class NoMask(_Mask):
    def __init__(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def prune(self):
        pass
    
    def regrow(self):
        pass

    def apply(self):
        pass

    def suppress_grad(self):
        pass

    def _criterion(self, *params) -> Odict:
        return Odict({name: torch.ones_like(param).bool() for name, param in params})

    def get_nonzero_weights_count(self):
        return self.numel()

    def get_zero_weights_count(self):
        return 0

    def numel(self):
        return 0

    def get_mask_sparsity(self):
        return 0.0

    def get_model_sparsity(self):
        return 0.0

    def names(self):
        return []

    def regenerate(self, regeneration_mask:Odict):
        pass

    def __getitem__(self, key):
        pass
    
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass