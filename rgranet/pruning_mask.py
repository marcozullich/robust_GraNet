from typing import Collection, List, Union
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
        device:Union[str, torch.device]=None,
    ):
        # scheduling_kwargs = coalesce(scheduling_kwargs, {"initial_pruning_rate": init_pruning_rate})
        # scheduling_kwargs["pruning_frequency"] = pruning_frequency

        pruning_rate_schedule = coalesce(pruning_rate_schedule, schedule.NoPruningRateScheduling)
        self.p = init_pruning_rate
        self.params_to_prune = params_to_prune
        self.scheduling = pruning_rate_schedule(**scheduling_kwargs)
        self.is_global = is_global
        # self.pruning_frequency = pruning_frequency
        # self.regrowth_frequency = regrowth_frequency
        # self.step_start_prune = step_start_prune
        # self.step_start_regrow = step_start_regrow

        self.params_to_prune = coalesce(self.params_to_prune, net.parameters_names)
        self.effective_params_to_prune = []
        self.net = net
        for name, _ in net.filtered_named_parameters(self.params_to_prune):
            self.effective_params_to_prune.append(name)
        self.device = coalesce(device, next(iter(self.net.parameters())).device)
        self.mask = self._init_mask()
        

    def _init_mask(self):
        mask = Odict()
        for (name, param) in self.net.filtered_named_parameters(self.effective_params_to_prune):
            mask[name] = torch.ones_like(param).bool().to(self.device)
        return mask

    def apply(self):
        for (_, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask.items()):
            msk = msk.to(param.device)
            param.data *= msk
            msk = msk.to(self.device)

    def suppress_grad(self):
        for (name, param), (_, msk) in zip(self.net.filtered_named_parameters(self.effective_params_to_prune), self.mask.items()):
            msk = msk.to(param.grad.device)
            param.grad *= msk
            msk = msk.to(self.device)


    def _criterion(self, *params) -> Odict:
        raise NotImplementedError("This is an abstract class")

    def _update(self, is_global=None, pruning_rate=None):
        is_global = coalesce(is_global, self.is_global)
        if is_global:
            parameters = self.net.filtered_named_parameters(self.effective_params_to_prune)
            new_mask = self._criterion(*parameters, pruning_rate=pruning_rate)
        else:
            new_mask = Odict()
            for (name, param) in self.net.filtered_named_parameters(self.effective_params_to_prune):
                new_mask[name] = self._criterion((name, param), pruning_rate=pruning_rate)[name]
        # self.mask_delta = Odict({n: m.logical_xor(new_mask) for n, m in self.mask.items()})
        # prev_sparsity = self.get_mask_sparsity()
        self.mask = new_mask
        # curr_sparsity = self.get_mask_sparsity()
        # self.delta_sparsity = curr_sparsity - prev_sparsity

    def prune(self, pruning_rate=None, is_global=None):
        is_global = coalesce(is_global, self.is_global)
        pruning_rate = coalesce(pruning_rate, self.p)
        if pruning_rate > 0.0:
            self._update(is_global, pruning_rate=pruning_rate)
            self.apply()
            # print(f"\tsparsity after pruning {self.get_mask_sparsity():.6f}")

    def step(self):
        self.scheduling.step()
        self.p = self.scheduling.current_pruning_rate
    
    def regrow(self):
        if self.scheduling.regrowth_rate > 0.0:
            regen_mask = gradient_based_neuroregeneration(self.net, self.params_to_prune, self.scheduling.regrowth_rate, self.is_global)
            self.regenerate(regen_mask)
            # print(f"\tsparsity after regrowth {self.get_mask_sparsity():.6f}")
    
    def get_nonzero_weights_count_per_module(self) -> List:
        return {n: torch.nonzero(msk).size(0) for (n, msk) in self.mask.items()}
 
    def get_nonzero_weights_count(self):
        return sum(self.get_nonzero_weights_count_per_module().values())
    
    def get_zero_weights_count_per_module(self) -> List:
        return {n: (m.numel() - z) for (n, m), (_, z) in zip(self.mask.items(), self.get_nonzero_weights_count_per_module().items())}

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
    
    def to(self, device:Union[str, torch.device]):
        for (_, msk) in self.mask.items():
            msk = msk.to(device)

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
    def _criterion(self, *params, pruning_rate=None) -> Odict:
        pruning_rate = coalesce(pruning_rate, self.p)
        flattened_abs_params = torch.cat([param[self.mask[name]].abs() for name, param in params])
        index = int(pruning_rate * flattened_abs_params.numel())
        # if index is 0, the pruning rate is too small to prune anything
        pth_quantile = None
        if index > 0:
            # print(f"PRUNE: {pruning_rate:.6f} ++ {self.scheduling.regrowth_rate:.6f} - num params {flattened_abs_params.numel()} - index {index} - spa {self.get_mask_sparsity():.6f}")
            pth_quantile = flattened_abs_params.kthvalue(index)[0]
            msk = Odict({name: (param.abs() >= pth_quantile).to(self.device) for name, param in params})
        else:
        # # print(f"PRUNE: {pruning_rate} - num params {flattened_abs_params.numel()} - index {index} ---")
            msk = Odict({name: (self.mask[name]).to(self.device) for name, _ in params})
        
        return msk

class RGraNetMask(LMMask):
    def __init__(
        self,
        init_pruning_rate:float,
        net:torch.nn.Module,
        tot_num_pruning_ite:int,
        params_to_prune:Collection=None,
        initial_sparsity:float=0.0,
        final_sparsity:float=.9,
        initial_ite_pruning:int=0,
        pruning_frequency:int=20,
        regrowth_to_prune_ratio:float=1.0,
    ):
        super().__init__(
            init_pruning_rate=init_pruning_rate,
            net=net,
            params_to_prune=params_to_prune,
            pruning_rate_schedule=schedule.PruningRateCubicSchedulingWithRegrowth,
            scheduling_kwargs={
                "initial_sparsity": initial_sparsity,
                "final_sparsity": final_sparsity,
                "initial_ite_pruning": initial_ite_pruning,
                "pruning_frequency": pruning_frequency,
                "tot_num_pruning_ite": tot_num_pruning_ite,
                "regrowth_to_prune_ratio": regrowth_to_prune_ratio,
            },
            is_global=True,
        )

class GradualPruningMask(LMMask):
    def __init__(
        self,
        init_pruning_rate:float,
        net:torch.nn.Module,
        tot_num_pruning_ite:int,
        params_to_prune:Collection=None,
        initial_sparsity:float=0.0,
        final_sparsity:float=.9,
        initial_ite_pruning:int=0,
        pruning_frequency:int=20,
    ):
        super().__init__(
            init_pruning_rate=init_pruning_rate,
            net=net,
            params_to_prune=params_to_prune,
            pruning_rate_schedule=schedule.PruningRateCubicScheduling,
            scheduling_kwargs={
                "initial_sparsity": initial_sparsity,
                "final_sparsity": final_sparsity,
                "initial_ite_pruning": initial_ite_pruning,
                "pruning_frequency": pruning_frequency,
                "tot_num_pruning_ite": tot_num_pruning_ite
            },
            is_global=True,
        )
    
    def regrow(self):
        pass

class GraNetMask(LMMask):
    def __init__(
        self,
        init_pruning_rate:float,
        net:torch.nn.Module,
        tot_num_pruning_ite:int,
        params_to_prune:Collection=None,
        initial_sparsity:float=0.0,
        final_sparsity:float=.9,
        initial_ite_pruning:int=0,
        pruning_frequency:int=20,
        initial_death_and_regrowth_rate:float=0.5
    ):
        super().__init__(
            init_pruning_rate=init_pruning_rate,
            net=net,
            params_to_prune=params_to_prune,
            pruning_rate_schedule=schedule.PruningRateCubicScheduling,
            scheduling_kwargs={
                "initial_sparsity": initial_sparsity,
                "final_sparsity": final_sparsity,
                "initial_ite_pruning": initial_ite_pruning,
                "pruning_frequency": pruning_frequency,
                "tot_num_pruning_ite": tot_num_pruning_ite
            },
            is_global=True,
        )
        self.secondary_scheduler = schedule.PruningRateCosineScheduling(
            initial_pruning_rate=initial_death_and_regrowth_rate,
            tot_num_pruning_ite=tot_num_pruning_ite,
            initial_ite_pruning=initial_ite_pruning,
            pruning_frequency=pruning_frequency
        )

    def prune(self):
        # prune is composed of two parts
        # 1. main pruning: prune the weights globally according to the main scheduler
        super().prune()
        # 2. magnitude death: prune the weights locally for each layer according to the secondary scheduler
        if self.allow_death_and_regrowth_phase():
            mask_statistics_before = self.get_zero_weights_count_per_module()
            super().prune(pruning_rate=self.r, is_global=False)
            mask_statistics_after = self.get_zero_weights_count_per_module()
            # prepare for regrowth
            self.num_params_to_regrow = {n: (mask_statistics_after[n] - m_bef) for n, m_bef in mask_statistics_before.items()}
            

    def allow_death_and_regrowth_phase(self):
        return self.r > 0.0 and self.p > 0.0 and not self.scheduling.has_ended()

    def step(self):
        self.scheduling.step()
        
        current_density = 1 - self.get_mask_sparsity()
        target_density = 1 - self.scheduling.current_sparsity
        self.p = 1 - target_density / current_density

        self.secondary_scheduler.step()
        self.r = self.secondary_scheduler.current_pruning_rate
    
    def regrow(self):
        # regrow is local and applied only to the weights pruned in this ite, hence the need for determining a mask_delta
        if self.allow_death_and_regrowth_phase():
            regen_mask = gradient_based_neuroregeneration(self.net, self.effective_params_to_prune, regrowth_rate=None, is_global=False, num_to_regrow=self.num_params_to_regrow)
            self.regenerate(regen_mask)
    
    def _update(self, is_global, pruning_rate=None):
        # self.old_mask = Odict({k: v.clone() for k, v in self.mask.items()})
        super()._update(is_global=is_global, pruning_rate=pruning_rate)


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