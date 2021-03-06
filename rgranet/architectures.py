import torch
from torch import nn
import timm
import torch.nn.functional as F
from .utils import coalesce



def get_model(name, num_classes, **kwargs):
    NET_PARSER = {
        "FCN4": FCN4,
        "PreActResNet18": PreActResNet18
    }
    if NET_PARSER.get(name) is not None:
        return NET_PARSER[name](num_classes=num_classes, **kwargs)
    kwargs["pretrained"] = coalesce(kwargs.get("pretrained"), False)
    return timm.create_model(name, num_classes=num_classes, **kwargs)

class FCN4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.m = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.m(x)

class PreActBlock(nn.Module):
    '''
    from basenet: https://github.com/bkj/basenet/
    '''
    def __init__(self, in_channels, out_channels, stride=1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    '''
    from basenet: https://github.com/bkj/basenet/
    '''
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        
        self.in_channels = 64
        
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1, seed=seed),
            self._make_layer(64, 128, num_blocks[1], stride=2, seed=seed),
            self._make_layer(128, 256, num_blocks[2], stride=2, seed=seed),
            self._make_layer(256, 256, num_blocks[3], stride=2, seed=seed),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, **kwargs):
        
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride, **kwargs))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x = x.half()
        x = self.prep(x)
        
        x = self.layers(x)
        
        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)
        
        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)
        
        x = torch.cat([x_avg, x_max], dim=-1)
        
        x = self.classifier(x)
        
        return x