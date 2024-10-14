import torch
from enum import Enum, IntEnum


class Precision(IntEnum):
    BINARY = 0
    TINARY = 1
    INT2 = 2
    INT3 = 3
    INT4 = 4
    INT5 = 5
    INT6 = 6
    INT7 = 7
    INT8 = 8
    INT9 = 9
    INT10 = 10
    FP16 = 16
    FP32 = 32


PRECISION_TO_BIT = {
    Precision.BINARY: 1,
    Precision.INT2: 2,
    Precision.INT3: 3,
    Precision.INT4: 4,
    Precision.INT5: 5,
    Precision.INT6: 6,
    Precision.INT7: 7,
    Precision.INT8: 8,
    Precision.INT9: 9,
    Precision.INT10: 10,
    Precision.FP16: 16,
    Precision.FP32: 32
}


# convert precision to str
PRECISION_TO_STR = {
    Precision.BINARY: 'binary',
    Precision.TINARY: 'tinary',
    Precision.INT2: 'int2',
    Precision.INT3: 'int3',
    Precision.INT4: 'int4',
    Precision.INT5: 'int5',
    Precision.INT6: 'int6',
    Precision.INT7: 'int7',
    Precision.INT8: 'int8',
    Precision.INT9: 'int9',
    Precision.INT10: 'int10',
    Precision.FP16: 'fp16',
    Precision.FP32: 'fp32'
}


STR_TO_PRECISION = {
    'binary': Precision.BINARY,
    'tinary': Precision.TINARY,
    'int2': Precision.INT2,
    'int3': Precision.INT3,
    'int4': Precision.INT4,
    'int5': Precision.INT5,
    'int6': Precision.INT6,
    'int7': Precision.INT7,
    'int8': Precision.INT8,
    'fp16': Precision.FP16,
    'fp32': Precision.FP32
}

INT_TO_PRECISION = {
    1: Precision.BINARY,
    2: Precision.INT2,
    3: Precision.INT3,
    4: Precision.INT4,
    5: Precision.INT5,
    6: Precision.INT6,
    7: Precision.INT7,
    8: Precision.INT8,
    16: Precision.FP16,
    32: Precision.FP32
}

'''
Basic quantizer layers
'''
class QuantizedModule(torch.nn.Module):
    def __init__(self, core, offload='cpu', name='') -> None:
        super().__init__()
        self.core = core
        self.hook_func = []
        self.registered_hook = []
        self.quantizer = []
        self.status = 'initialized'
        self.default_quantizer = None
        self.offload = offload
        self.name = name

        # set essential values for SDXL   
        if isinstance(self.core, torch.nn.Linear):
            setattr(self,"in_features", self.core.in_features)

    def to(self, device):
        if isinstance(device, (torch.device, str)):
            if self.default_quantizer is not None:
                self.default_quantizer.to(device)
        super().to(device)
        return self

    def register_quantizer(self, quantizer):
        if isinstance(quantizer, (list, tuple)):
            self.quantizer.extend(quantizer)
        else:
            self.quantizer.append(quantizer)
        return self
    
    def prepare_hook(self, load_hook_from_quantizers=True):
        if load_hook_from_quantizers:
            for quantizer in self.quantizer:
                quantizer.add_hook()

        self.registered_hook = []
        for hook in self.hook_func:
            self.registered_hook.append(self.core.register_forward_hook(hook))
    
    def remove_hook(self):
        for hook in self.registered_hook: hook.remove()
    
    def forward(self, x, scale=0):
        if self.status == "just_core":
            output = self.core(x)
            return output
        if self.status != 'quantized' or self.default_quantizer is None:
            output = self.core(x)
            for quantizer in self.quantizer:
                if quantizer.abit not in [Precision.FP16, Precision.FP32]:
                    if hasattr(quantizer, "observe"):
                        quantizer.observe(x)
                    else:
                        quantizer.a_observer = quantizer.a_observer.to(x.device)
                        quantizer.a_observer(x)
            return output
        return self.default_quantizer(x)

    def quantize(self):
        for quantizer in self.quantizer:
            quantizer.quantize()

    def set_default_quantizer(self, idx):
        origin_quantizer = self.default_quantizer
        if idx is None:
            self.default_quantizer = None
        else:
            self.default_quantizer = self.quantizer[idx]
            self.default_quantizer.to('cuda')

        if origin_quantizer is not None and origin_quantizer != self.default_quantizer:
            origin_quantizer.to(self.offload)
        self.status = 'quantized'
        return self