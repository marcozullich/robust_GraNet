import torch
from rgranet.pruning_mask import LMMask
from rgranet.model import Model
import rgranet.pruning_rate_schedule as A
from rgranet.neuroregeneration import gradient_based_neuroregeneration

torch.manual_seed(111)

class CustomLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Linear(3, 3)
        self.x.load_state_dict(
            {"weight": torch.arange(9).reshape(3, 3),
            "bias": torch.Tensor([10, 11, 12])
        })
        self.y = torch.nn.Linear(3, 3)
        self.y.load_state_dict(
            {"weight": torch.Tensor([[0,0,0],[1,2,3],[4,5,6]]),
            "bias": torch.Tensor([10, 11, 12])
        })
        self.x.requires_grad = True
        self.y.requires_grad = True
    def forward(self, x):
        return self.x(self.y(x))

# '''net = Model(torch.nn.Sequential(
#     torch.nn.Linear(3,5),
#     torch.nn.BatchNorm1d(5),
#     torch.nn.Linear(5,2),
# ),'''

# net = Model(CustomLinear(),
# mask_class=LMMask,
# init_pruning_rate=.5, mask_kwargs={"params_to_prune": ["weight"]})
# for n,p in net.named_parameters():
#     print(n, p.shape)
# m = net.mask

# m.step()

# y = net.forward(torch.randn(10,3)).sum() * 4
# y.backward()

# regenerated_params = gradient_based_neuroregeneration(net, ["weight"], .25)



# m.regenerate(regenerated_params)
# m

freq = 50
ites = 391

current_density = 100
p = A.PruningRateCubicSchedulingWithRegrowth(
    initial_sparsity=0,
    final_sparsity=.9,
    pruning_frequency=50,
    tot_num_pruning_ite=391
)
for i in range(0, freq * ites):
    p.step()
    if i % freq == 0 :
        density_after_pruning = current_density * (1-p.current_pruning_rate)
        density_after_regrowth = (current_density - density_after_pruning) * p.regrowth_rate + density_after_pruning
        print(f"{i} - p {p.current_pruning_rate:.5f} - dens {current_density:.5f} - after pru {density_after_pruning:.5f} - r {p.regrowth_rate} - after reg {density_after_regrowth}")
        current_density = density_after_regrowth
    # print(i, p.current_sparsity, p.current_pruning_rate, p.regrowth_rate)
p