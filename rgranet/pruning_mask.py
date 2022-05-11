from curses import raw
from typing import Collection, List, Union
from collections import OrderedDict as Odict
import torch
from enum import Enum
import math

from . import pruning_rate_schedule as schedule
from .utils import coalesce
from .neuroregeneration import gradient_based_neuroregeneration

class GradientsAccumulationMethod(Enum):
    NEVER = 0
    ALWAYS = 1
    BETWEEN_PRUNE_AND_REGROWTH = 2

class RGraNetMaskWhenPrune(Enum):
    PRUNE_BEFORE_FORWARD_REGROW_AFTER_BACKWARD = 0
    PRUNE_AND_REGROW_AFTER_UPDATE = 1

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
        initial_density:float=1.0,
        sparse_power_scale:float=1.0,
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
        # ###
        # print("Effective params to prune", self.effective_params_to_prune)
        # ###
        self.device = coalesce(device, next(iter(self.net.parameters())).device)
        self.initial_density = initial_density
        self.mask = self._init_mask(sparse_density=self.initial_density, sparse_power_scale=sparse_power_scale)
        self.apply()
        

    def _init_mask(self, sparse_density:float=1.0, sparse_power_scale=1.0):
        mask = Odict()
        for (name, param) in self.net.filtered_named_parameters(self.effective_params_to_prune):
            mask[name] = torch.ones_like(param).bool().to(self.device)
        if sparse_density < 1.0:
            self.sparse_init(power_scale=sparse_power_scale, density_0=sparse_density, dense_mask=mask)
        return mask

    def sparse_init(self, power_scale:float, density_0:float, dense_mask):
        assert density_0 > 0.0 and density_0 <= 1.0, f"density_0 must be between 0 and 1, got {density_0}"
        dense_layers = set()
        max_weighted_probability = math.inf

        while max_weighted_probability > 1.0:
            divisor = 0
            rhs = 0
            raw_probabilities = Odict()

            for name, mask in dense_mask.items():
                if name in dense_layers:
                    rhs -= mask.numel() * (1 - density_0)
                else:
                    rhs += mask.numel() * density_0
                    raw_probabilities[name] = (sum(mask.shape) / mask.numel()) ** power_scale
                    divisor += raw_probabilities[name] * mask.numel()
            
            eps = rhs / divisor
            max_probability, max_probability_idx = torch.Tensor(list(raw_probabilities.values())).topk(1)
            max_weighted_probability = (max_probability * eps).item()

            if max_weighted_probability > 1.0:
                dense_layers.add(list(raw_probabilities.keys())[max_probability_idx[0].item()])

        for name, mask in dense_mask.items():
            if name not in dense_layers:
                dense_mask[name] = torch.rand(mask.shape) < raw_probabilities[name] * eps

                        






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
            # ###
            # parameters = list(parameters)
            # print("filtered parameters\n", parameters)
            # ###
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
            print(f"\tsparsity after pruning {self.get_mask_sparsity():.6f}")

    def step(self):
        self.scheduling.step()
        self.p = self.scheduling.current_pruning_rate
    
    def regrow(self, named_gradients=None):
        if self.scheduling.can_regrow and self.scheduling.regrowth_rate > 0.0:
            regen_mask = gradient_based_neuroregeneration(self.net, self.params_to_prune, self.scheduling.regrowth_rate, is_global=self.is_global, named_gradients=named_gradients)
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
    
    def load_mask(self, mask_dict, apply=True):
        self.mask = mask_dict["mask"]
        print("Loaded mask", end=" ")
        if apply:
            self.apply()
            print("and applied to model.")
        else:
            print("")
        
class LMMask(_Mask):
    def _criterion(self, *params, pruning_rate=None) -> Odict:
        pruning_rate = coalesce(pruning_rate, self.p)
        
        flattened_abs_params = torch.cat([param[self.mask[name]].abs() for name, param in params])
        # ###
        # print("Mask names:", self.mask.keys())
        # flattened_abs_params = []
        # print("Param names:")
        # for name, param in params:
        #     print(name)
        #     flattened_abs_params.append(param[self.mask[name]].abs())
        # flattened_abs_params = torch.cat(flattened_abs_params)
        # ###
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
        # regrowth_to_prune_ratio:float=1.0,
        regrowth_delay:int=0,
        # accumulate_gradients_before_regrowth:bool=False,
        gradients_accumulation_method = GradientsAccumulationMethod.NEVER,
        death_and_regrowth_rate:int=1,
        death_and_regrowth_global:bool=True,
        when_prune=RGraNetMaskWhenPrune.PRUNE_BEFORE_FORWARD_REGROW_AFTER_BACKWARD
    ):
        super().__init__(
            init_pruning_rate=init_pruning_rate,
            net=net,
            params_to_prune=params_to_prune,
            pruning_rate_schedule=schedule.PruningRateCubicSchedulingWithFixedRegrowth,
            initial_density=(1-initial_sparsity),
            scheduling_kwargs={
                "initial_sparsity": initial_sparsity,
                "final_sparsity": final_sparsity,
                "initial_ite_pruning": initial_ite_pruning,
                "pruning_frequency": pruning_frequency,
                "tot_num_pruning_ite": tot_num_pruning_ite,
                # "regrowth_to_prune_ratio": regrowth_to_prune_ratio,
                "p_regen": death_and_regrowth_rate,
                "initial_ite_regrow": initial_ite_pruning + regrowth_delay,
                "regrowth_frequency": pruning_frequency,
            },
            is_global=True,
        )
        # self.accumulate_gradients_before_regrowth = accumulate_gradients_before_regrowth
        self.gradients_accumulation_method = gradients_accumulation_method
        self.death_and_regrowth_global = death_and_regrowth_global
        self.need_gradient_reset = False
    
    def prune(self):
        if self.death_and_regrowth_global:
            num_params_before = self.get_nonzero_weights_count()
            num_params_after_fase_1_pruning = int(num_params_before * (1 - self.scheduling.fase_1_pruning_rate))
            super().prune()
            if self.p > 0.0:
                num_params_after_fase_2_pruning = self.get_nonzero_weights_count()
                print(f"## {self.scheduling.step_counter} - Pruned with rate {self.p:.6f} (n. params {num_params_before - num_params_after_fase_2_pruning})")
                self.num_params_to_regrow = num_params_after_fase_1_pruning - num_params_after_fase_2_pruning
        else:
            super().prune(pruning_rate=self.scheduling.fase_1_pruning_rate, is_global=True)
            if self.p > 0.0:
                num_zeros_after_fase_1 = self.get_nonzero_weights_count_per_module()
                super().prune(pruning_rate=self.scheduling.secondary_scheduler.current_pruning_rate, is_global=False)
                num_zeros_after_fase_2 = self.get_nonzero_weights_count_per_module()
                self.num_params_to_regrow = {name: num_zeros_after_fase_1[name] - num_zeros_after_fase_2[name] for name in num_zeros_after_fase_1.keys()}


    
    def regrow(self, named_gradients=None):
        if hasattr(self, "num_params_to_regrow"):
            if isinstance(self.num_params_to_regrow, int):
                regrow_num_positive = True
            elif isinstance(self.num_params_to_regrow, dict):
                regrow_num_positive = any(num > 0 for num in self.num_params_to_regrow.values())
            else:
                raise ValueError(f"Regrowth num params must be int or dict, got {type(self.num_params_to_regrow)}")
            if self.scheduling.can_regrow and regrow_num_positive:
                if isinstance(self.num_params_to_regrow, int): 
                    print(f"## {self.scheduling.step_counter} - Regrowing {self.num_params_to_regrow} params")
                regen_mask = gradient_based_neuroregeneration(self.net, self.effective_params_to_prune, regrowth_rate=None, num_to_regrow=self.num_params_to_regrow, is_global=self.death_and_regrowth_global, named_gradients=named_gradients, mask=self)
                
                self.regenerate(regen_mask)
                print("After regrow:", self.get_mask_sparsity())
                if self.gradients_accumulation_method != GradientsAccumulationMethod.ALWAYS:
                    self.need_gradient_reset = True

        
            
            

        


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
    
    def regrow(self, *args, **kwargs):
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
        return self.r > 0.0 and self.p > 0.0 and not self.scheduling.has_ended_prune()

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