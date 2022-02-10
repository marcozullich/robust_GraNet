import torch

from .utils import coalesce

class CyclicLRWithBurnout(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer:torch.optim.Optimizer,
        base_lr:float,
        max_lr:float,
        step_size_up:int,
        step_size_down:int=None,
        total_steps:int=None,
        min_lr_down:float=None,
        
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = coalesce(step_size_down, step_size_up)
        self.cycle_length = self.step_size_up + self.step_size_down
        self.total_steps = total_steps
        self.min_lr_down = coalesce(min_lr_down, base_lr)
        self.step_counter = -1
        self.step()
    
    def get_lr(self) -> float:
        if self.total_steps is not None and self.step_counter >= self.total_steps:
            # burnout
            return self.min_lr_down * 0.1
        if (self.step_counter % self.cycle_length) < self.cycle_length / 2:
            # ascending
            return self.base_lr + (self.max_lr - self.base_lr) * self.step_counter / self.step_size_up
        # descending
        delta_lr = self.max_lr - self.min_lr_down
        step_ratio = self.step_size_up / self.step_size_down
        return self.max_lr + delta_lr * step_ratio - delta_lr / self.step_size_down * self.step_counter

    def set_lr(self) -> None:
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self) -> None:
        self.step_counter += 1
        self.set_lr()
    
    def state_dict(self) -> dict:
        return {
            "step_counter": self.step_counter,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.step_counter = state_dict["step_counter"]
        self.set_lr()

    def get_last_lr(self) -> float:
        return [pg["lr"] for pg in self.optimizer.param_groups]