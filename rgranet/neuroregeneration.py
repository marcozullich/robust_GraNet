from multiprocessing.sharedctypes import Value
from this import d
import torch
from typing import Collection, Dict, Union
from collections import OrderedDict as Odict

from .utils import coalesce


def _neuroregenerate_params(
    mask: Odict,
    regrowth_rate: float,
    *named_gradients: Collection,
    device: Union[str, torch.device] = None,
    num_to_regrow: int = None
) -> Odict:
    pruned_params_abs_grad = torch.cat([g[~mask[name]].abs() for name, g in named_gradients])
    if num_to_regrow is None:
        grad_threshold_index = int(pruned_params_abs_grad.numel() * (1-regrowth_rate))
    else:
        grad_threshold_index = pruned_params_abs_grad.numel() - num_to_regrow
    if grad_threshold_index > 0:
        quantile = pruned_params_abs_grad.kthvalue(grad_threshold_index).values
    # print(f"REGROW: {regrowth_rate:.6f} - num params {pruned_params_abs_grad.numel()} - index {grad_threshold_index} - num params {pruned_params_abs_grad.numel()}")
        regenerated_params = Odict({n: (g.abs() > quantile).to(device) for n, g in named_gradients})
    elif grad_threshold_index == 0:
        regenerated_params = Odict({n: torch.ones_like(g).bool().to(device) for n, g in named_gradients})
    else:
        raise ValueError(f"Got index {grad_threshold_index} less than 0")
    return regenerated_params


def gradient_based_neuroregeneration(
    net:torch.nn.Module,
    params_to_prune:Collection,
    regrowth_rate:float=None,
    num_to_regrow:Union[int, Dict[str, int]]=None,
    is_global:bool=False,
    mask:Odict=None,
    device:Union[str, torch.device]=None
) -> Odict:
    regrowth_rate, num_to_regrow = validate_parse_args(regrowth_rate, num_to_regrow, is_global, len(params_to_prune))

    if num_to_regrow is not None:
        assert all(n>=0 for n in num_to_regrow.values()), f"All numbers to regrow must be positive. Found {[(k, n) for k, n in num_to_regrow.items() if n<0]}"

    named_gradients = ((n, p.grad) for n, p in net.filtered_named_parameters(params_to_prune))
    mask = coalesce(mask, net.mask)
    device = coalesce(device, mask.device)
    if is_global:
        regenerated_params = _neuroregenerate_params(net.mask, regrowth_rate, *named_gradients, device=device, num_to_regrow=num_to_regrow)
    else:
        regenerated_params = {}
        for name, grad in named_gradients:
            regenerated_params[name] = _neuroregenerate_params(net.mask, regrowth_rate, (name, grad), device=device, num_to_regrow=num_to_regrow[name])[name]

    return regenerated_params
    
def validate_parse_args(regrowth_rate:int, num_to_regrow:Union[int, Dict[str, int]], is_global:bool, num_params_to_prune:int):
    if regrowth_rate is None and num_to_regrow is None:
        raise ValueError("Either regrowth_rate or num_to_regrow must be specified")
    elif regrowth_rate is not None and num_to_regrow is not None:
        raise ValueError("Only one of regrowth_rate or num_to_regrow can be specified")
    else:
        if regrowth_rate is None:
            if is_global:
                if isinstance(num_to_regrow, dict):
                    raise ValueError("If global regrowth is specified, num_to_regrow must be a single value")
            else:
                if len(num_to_regrow) != num_params_to_prune:
                    raise ValueError("If num_to_regrow is a list, its size must be equal to the number of blocks to prune")
    
    return regrowth_rate, num_to_regrow

    

    