import torch
from typing import Collection
from collections import OrderedDict as Odict

def _neuroregenerate_params(
    mask: Odict,
    regrowth_rate: float,
    *named_gradients: Collection
) -> Odict:                           
    pruned_params_abs_grad = torch.cat([g[~mask[name]].abs() for name, g in named_gradients])
    grad_threshold_index = int(pruned_params_abs_grad.numel() * (1-regrowth_rate))
    quantile = pruned_params_abs_grad.kthvalue(grad_threshold_index).values
    # print(f"REGROW: {regrowth_rate:.6f} - num params {pruned_params_abs_grad.numel()} - index {grad_threshold_index} - num params {pruned_params_abs_grad.numel()}")
    regenerated_params = Odict({n: g.abs() > quantile for n, g in named_gradients})
    return regenerated_params


def gradient_based_neuroregeneration(net:torch.nn.Module,
                                     params_to_prune:Collection,
                                     regrowth_rate:float,
                                     is_global:bool=False
                                     ) -> Odict:
    named_gradients = ((n, p.grad) for n, p in net.filtered_named_parameters(params_to_prune))
    if is_global:
        regenerated_params = _neuroregenerate_params(net.mask, regrowth_rate, *named_gradients)
    else:
        regenerated_params = {}
        for name, grad in named_gradients:
            regenerated_params[name] = _neuroregenerate_params(net.mask.mask_delta, regrowth_rate, (name, grad))[name]

    return regenerated_params
    


    

    