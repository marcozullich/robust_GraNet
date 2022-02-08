import torch
from typing import Union

DTYPE_PARSER = {	
    'float32': torch.float32,
    'float': torch.float,
    'float64': torch.float64,
    'double': torch.double,
    'complex64': torch.complex64,
    'cfloat': torch.cfloat,
    'complex128': torch.complex128,
    'cdouble': torch.cdouble,
    'float16': torch.float16,
    'half': torch.half,
    'bfloat16': torch.bfloat16,
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.short,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.long,
    'bool': torch.bool,
}	


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def accuracy(preds, labels):
    return (correct_preds(preds, labels) / len(labels)) 


def set_dtype_(tensor_or_module:Union[torch.Tensor, torch.nn.Module], dtype:Union[torch.dtype,str]):
    """
    Set the dtype of a tensor or module.
    """
    if isinstance(dtype, str):
        dtype = DTYPE_PARSER[dtype]
    tensor_or_module.type(dtype)