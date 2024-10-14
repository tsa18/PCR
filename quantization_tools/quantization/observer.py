import torch
import torch.nn as nn
class HistogramObserver(nn.Module):
    def __init__(self,momentum=0.1,percentile=0.9999):
        super(HistogramObserver,self).__init__()
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.register_buffer("min_val", torch.zero((0),dtype=torch.float32))
        self.register_buffer("max_val", torch.zero((0),dtype=torch.float32))

    @torch.no_grad()
    def forward(self,input):
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(
            input.abs().view(-1), int(self.percentile*intput.view(-1).size(0)), dim=0
        )[0]
        min_val_cur = torch.min(input)
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
            min_val = min_val_cur
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        self.max_val.copy_(max_val)
        self.min_val.copy_(min_val)


