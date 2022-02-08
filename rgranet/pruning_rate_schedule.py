from hashlib import new
import math


class _PruningRateScheduling:
    def __init__(self, initial_pruning_rate):
        self.initial_pruning_rate = initial_pruning_rate
        self.current_pruning_rate = initial_pruning_rate
        self.regrowth_rate = None

    def step(self):
        raise NotImplementedError("This is an abstract class")

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
        return True

class NoPruningRateScheduling(_PruningRateScheduling):
    def step(self):
        return self.current_pruning_rate
    

class PruningRateCosineScheduling(_PruningRateScheduling):
    def __init__(self, initial_pruning_rate, tot_num_ite, pruning_rate_min=5e-3):
        super().__init__(initial_pruning_rate)
        self.T_max = tot_num_ite
        self.eta_min = pruning_rate_min
        self.step_counter = 0
    
    def step(self):
        self.current_pruning_rate = self.eta_min + .5 * (self.current_pruning_rate - self.eta_min) * (1 + math.cos(math.pi * self.step_counter / self.T_max))
        self.step_counter += 1
    
    def state_dict(self):
        return {
            "current_pruning_rate": self.current_pruning_rate,
            "step_counter": self.step_counter
        }
    


class PruningRateLinearScheduling(_PruningRateScheduling):
    def __init__(self, initial_pruning_rate, final_pruning_rate, tot_num_ite):
        super().__init__(initial_pruning_rate)
        self.final_pruning_rate = final_pruning_rate
        self.T_max = tot_num_ite
        self.update_step = (self.final_pruning_rate - self.initial_pruning_rate) / self.T_max
    
    def step(self):
        self.current_pruning_rate -= self.update_step

class PruningRateCubicScheduling(_PruningRateScheduling):
    def __init__(self, initial_sparsity:float, final_sparsity:float, tot_num_pruning_ite:int, initial_ite_pruning:int=0, pruning_frequency:int=1):
        super().__init__(0)
        self.initial_sparsity = initial_sparsity
        self.current_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.T_max = tot_num_pruning_ite
        self.pruning_frequency = pruning_frequency
        self.initial_ite_pruning = initial_ite_pruning
        self.update_denominator = self.T_max * self.pruning_frequency
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
        self.step_counter += 1
    
    def state_dict(self):
        return {
            "current_pruning_rate": self.current_pruning_rate,
            "current_sparsity": self.current_sparsity,
            "step_counter": self.step_counter
        }
    
    def can_prune(self):
        return self.step_counter >= self.initial_ite_pruning and self.step_counter <= (self.T_max * self.pruning_frequency + self.initial_ite_pruning) and ((self.step_counter + self.initial_ite_pruning) % self.pruning_frequency == 0)
    
class PruningRateCubicSchedulingWithRegrowth(PruningRateCubicScheduling):
    def __init__(self, initial_sparsity:float, final_sparsity:float, tot_num_pruning_ite:int, initial_ite_pruning:int=0, pruning_frequency:int=1):
        super().__init__(initial_sparsity, final_sparsity, tot_num_pruning_ite, initial_ite_pruning=initial_ite_pruning, pruning_frequency=pruning_frequency)
        self.regrowth_rate = 0.0
    
    def step(self):
        prev_density =  1 - self.current_sparsity
        super().step()
        target_density = 1 - self.current_sparsity
        if prev_density > target_density:
            delta = 4 * prev_density * target_density - 3 * prev_density * prev_density
            if delta < 0:
                raise ValueError(f"Delta is negative. Target density ({target_density:.4f}) should be ≥ ¾ × previous density ({prev_density:.4f}). Currently target density is {target_density/prev_density:.4f}. Try decreasing the final sparsity or increasing the numbe of pruning iterations.")

            corrected_pruning_rate = (prev_density - math.sqrt(delta)) / (2 * prev_density)
            self.current_pruning_rate = corrected_pruning_rate

            density_before_regrowth = prev_density * (1 - corrected_pruning_rate)

            self.regrowth_rate = (prev_density - density_before_regrowth) * corrected_pruning_rate / (1 - density_before_regrowth)
            # density_before_regrowth * corrected_pruning_rate / (1 - density_before_regrowth)
            

        else:
            self.regrowth_rate = 0.0
        
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["regrowth_rate"] = self.regrowth_rate
        return state_dict