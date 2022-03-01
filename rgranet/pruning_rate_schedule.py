import math

from rgranet.utils import coalesce


class _PruningRateScheduling:
    def __init__(self, initial_pruning_rate, tot_num_pruning_ite:int, pruning_frequency:int, initial_ite_pruning:int=0):
        self.initial_pruning_rate = initial_pruning_rate
        self.current_pruning_rate = initial_pruning_rate
        self.tot_num_pruning_ite = tot_num_pruning_ite
        self.initial_ite_pruning = initial_ite_pruning
        self.pruning_frequency = pruning_frequency
        self.regrowth_rate = None
        self.step_counter = 0
        

    def step(self):
        self.step_counter += 1

    def supports_regrowth(self):
        return self.regrowth_rate is not None
    
    def state_dict(self):
        return {
            "current_pruning_rate": self.current_pruning_rate,
        }
    
    def load_state_dict(self, state_dict):
        for attr, value in state_dict.items():
            setattr(self, attr, value)
    
    def can_prune(self):
        return self.step_counter >= self.initial_ite_pruning and (not self.has_ended_prune()) and ((self.step_counter + self.initial_ite_pruning) % self.pruning_frequency == 0)
    
    def has_ended_prune(self):
        return self.step_counter > (self.tot_num_pruning_ite * self.pruning_frequency + self.initial_ite_pruning)

class NoPruningRateScheduling(_PruningRateScheduling):
    def step(self):
        return self.current_pruning_rate
    

class PruningRateCosineScheduling(_PruningRateScheduling):
    def __init__(self, initial_pruning_rate, tot_num_pruning_ite, pruning_rate_min=5e-3, initial_ite_pruning=0, pruning_frequency=1):
        super().__init__(initial_pruning_rate, tot_num_pruning_ite, pruning_frequency, initial_ite_pruning)
        self.tot_num_pruning_ite = tot_num_pruning_ite
        self.pruning_rate_min = pruning_rate_min
        self.step_counter = 0
    
    def step(self):
        if self.can_prune():
            self.current_pruning_rate = self.pruning_rate_min + .5 * (self.current_pruning_rate - self.pruning_rate_min) * (1 + math.cos(math.pi * (self.step_counter // self.pruning_frequency) / self.tot_num_pruning_ite))
        super().step()
    
    def state_dict(self):
        return {
            "current_pruning_rate": self.current_pruning_rate,
            "step_counter": self.step_counter
        }
    


class PruningRateLinearScheduling(_PruningRateScheduling):
    def __init__(self, initial_pruning_rate, final_pruning_rate, tot_num_pruning_ite):
        super().__init__(initial_pruning_rate)
        self.final_pruning_rate = final_pruning_rate
        self.tot_num_pruning_ite = tot_num_pruning_ite
        self.update_step = (self.final_pruning_rate - self.initial_pruning_rate) / self.tot_num_pruning_ite
    
    def step(self):
        if self.can_prune():
            self.current_pruning_rate -= self.update_step

class PruningRateCubicScheduling(_PruningRateScheduling):
    def __init__(self, initial_sparsity:float, final_sparsity:float, tot_num_pruning_ite:int, initial_ite_pruning:int=0, pruning_frequency:int=1):
        super().__init__(
            initial_ite_pruning=initial_ite_pruning,
            tot_num_pruning_ite=tot_num_pruning_ite,
            pruning_frequency=pruning_frequency,
            initial_pruning_rate=0.0
        )
        self.initial_sparsity = initial_sparsity
        self.current_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.update_denominator = self.tot_num_pruning_ite * self.pruning_frequency
        self.delta_sparsity = self.final_sparsity - self.initial_sparsity
        self.step_counter = 0
        self.current_pruning_rate = 0.0
    
    def _get_pruning_rate(self, current_sparsity, new_sparsity):
        return 1 - ((1 - new_sparsity) / (1 - current_sparsity))

    def step(self):
        if self.can_prune():
            new_sparsity = self.final_sparsity - self.delta_sparsity * ((1 - (self.step_counter - self.initial_ite_pruning) / self.update_denominator) ** 3)
            self.current_pruning_rate = self._get_pruning_rate(self.current_sparsity, new_sparsity)
            self.current_sparsity = new_sparsity
        else:
            self.current_pruning_rate = 0.0
        super().step()
    
    def state_dict(self):
        return {
            "current_pruning_rate": self.current_pruning_rate,
            "current_sparsity": self.current_sparsity,
            "step_counter": self.step_counter
        }

class PruningRateCubicSchedulingWithFixedRegrowth(PruningRateCubicScheduling):
    def __init__(
        self,
        initial_sparsity:float,
        final_sparsity:float,
        tot_num_pruning_ite:int,
        initial_ite_pruning:int=0,
        pruning_frequency:int=1,
        p_regen:float=1.0,
        initial_ite_regrow:int=None,
        regrowth_frequency:int=None,
    ):
        initial_ite_regrow = coalesce(initial_ite_regrow, initial_ite_pruning)
        regrowth_frequency = coalesce(regrowth_frequency, pruning_frequency)
        super().__init__(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            initial_ite_pruning=initial_ite_pruning,
            pruning_frequency=pruning_frequency,
            tot_num_pruning_ite=tot_num_pruning_ite,
        )
        self.p_regen = p_regen
        self.initial_ite_regrow = initial_ite_regrow
        self.regrowth_frequency = regrowth_frequency
        self.step_counter = 0
        self.fase_1_pruning_rate = 0.0
        
    def can_regrow(self):
        return self.step_counter >= self.initial_ite_regrow and (not self.has_ended_regrow()) and ((self.step_counter + self.initial_ite_regrow) % self.regrowth_frequency == 0)

    def has_ended_regrow(self):
        return self.step_counter > (self.tot_num_pruning_ite * self.regrowth_frequency + self.initial_ite_regrow)
    
    def is_waiting_for_regrowth(self):
        if (self.step_counter - self.initial_ite_regrow) % self.regrowth_frequency == 0:
            return True
        
        current_regrow_phase = max(-1, (self.step_counter - self.initial_ite_regrow) // self.regrowth_frequency )
        current_prune_phase = max(-1, (self.step_counter - self.initial_ite_pruning) // self.pruning_frequency)
        next_regrow_ite = (current_regrow_phase + 1) * self.regrowth_frequency + self.initial_ite_regrow if current_regrow_phase > 0 else self.initial_ite_regrow

        return current_prune_phase > 0 and (next_regrow_ite - self.initial_ite_pruning) // self.pruning_frequency == current_prune_phase

    def step(self):
        prev_sparsity = self.current_sparsity
        waits_regrowth = self.is_waiting_for_regrowth()
        super().step()
        if self.current_sparsity > prev_sparsity:
            if waits_regrowth:
                target_density_before_regrowth = (1 - self.current_sparsity) * (1 - self.current_pruning_rate) * (1 - self.p_regen)
                self.fase_1_pruning_rate = self.current_pruning_rate
                self.current_pruning_rate = self._get_pruning_rate(prev_sparsity, 1 - target_density_before_regrowth)
            self.regrowth_rate = None # must regrow with int numbers


    
class PruningRateCubicSchedulingWithRegrowth(PruningRateCubicScheduling):
    def __init__(
        self,
        initial_sparsity:float,
        final_sparsity:float,
        tot_num_pruning_ite:int,
        initial_ite_pruning:int=0,
        pruning_frequency:int=1,
        regrowth_to_prune_ratio:float=1.0,
        initial_ite_regrow:int=None,
        regrowth_frequency:int=None,
):      
        initial_ite_regrow = coalesce(initial_ite_regrow, initial_ite_pruning)
        regrowth_frequency = coalesce(regrowth_frequency, pruning_frequency)

        super().__init__(initial_sparsity, final_sparsity, tot_num_pruning_ite, initial_ite_pruning=initial_ite_pruning, pruning_frequency=pruning_frequency)
        self.regrowth_rate = 0.0
        self.k = regrowth_to_prune_ratio

        self.initial_ite_regrow = initial_ite_regrow
        self.regrowth_frequency = regrowth_frequency
        
        self.has_pruned_and_not_regrown = False
    
    def can_regrow(self):
        return self.step_counter >= self.initial_ite_regrow and (not self.has_ended_regrow()) and ((self.step_counter + self.initial_ite_regrow) % self.regrowth_frequency == 0)

    def has_ended_regrow(self):
        return self.step_counter > (self.tot_num_pruning_ite * self.regrowth_frequency + self.initial_ite_regrow)

    def step(self):
        prev_density =  1 - self.current_sparsity
        super().step()
        target_density = 1 - self.current_sparsity
        if prev_density > target_density:
            delta = 4 * self.k * prev_density * target_density - (4 * self.k - 1) * prev_density * prev_density
            if delta < 0:
                raise ValueError(f"Delta is negative. Target density ({target_density:.4f}) should be ≥ ¾ × previous density ({prev_density:.4f}). Currently target density is {target_density/prev_density:.4f}. Try decreasing the final sparsity or increasing the numbe of pruning iterations.")

            corrected_pruning_rate = (prev_density - math.sqrt(delta)) / (2 * prev_density)
            self.current_pruning_rate = corrected_pruning_rate

            density_before_regrowth = prev_density * (1 - corrected_pruning_rate)

            self.regrowth_rate = (prev_density - density_before_regrowth) * self.k * corrected_pruning_rate / (1 - density_before_regrowth)
            # density_before_regrowth * corrected_pruning_rate / (1 - density_before_regrowth)
            

        else:
            self.regrowth_rate = 0.0

    def is_waiting_for_regrowth(self):
        current_regrow_phase = max(-1, (self.step_counter - self.initial_ite_regrow) // self.regrowth_frequency )
        current_prune_phase = max(-1, (self.step_counter - self.initial_ite_pruning) // self.pruning_frequency)
        next_regrow_ite = (current_regrow_phase + 1) * self.regrowth_frequency + self.initial_ite_regrow if current_regrow_phase > 0 else self.initial_ite_regrow

        return current_prune_phase > 0 and (next_regrow_ite - self.initial_ite_pruning) // self.pruning_frequency == current_prune_phase

        
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["regrowth_rate"] = self.regrowth_rate
        return state_dict


    
