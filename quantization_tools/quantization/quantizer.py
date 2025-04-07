from torch.ao.quantization.observer import (
    PerChannelMinMaxObserver,
    MinMaxObserver, 
    HistogramObserver,
    MovingAverageMinMaxObserver
)

import time
import math
import torch
import logging
import torch.nn.functional as F
from . import Precision, PRECISION_TO_BIT, PRECISION_TO_STR
from .loss import BrecqLoss, lp_loss, round_ste
from tqdm import tqdm
import numpy as np
import os 
import scipy.stats as stats

# we still need to remove the nn.Module because of the loop bug: hub -> quantizer -> hub
class BaseQuantizer:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# we need to remove uncessary hooks
def generate_track_input_output_hook(offload_device='cpu'):
    def track_input_output_hook(module, inputs, outputs):
        if not hasattr(module, 'input_output_tracks'):
            module.input_output_tracks = []
        if not torch.is_tensor(inputs):
            inputs = [inp.detach().to(offload_device) for inp in inputs]
        else:
            inputs = [inputs.detach().to(offload_device)]
        if not torch.is_tensor(outputs):
            outputs = [out.detach().to(offload_device) for out in outputs]
        else:
            outputs = [outputs.detach().to(offload_device)]
        module.input_output_tracks.append([inputs, outputs])
    return track_input_output_hook

track_input_output_hook_to_cpu = generate_track_input_output_hook('cpu')
track_input_output_hook_to_cuda = generate_track_input_output_hook('cuda')

def generate_track_grad_hook(offload_device='cpu'):
    def track_grad_hook(module, grad_input, grad_output):
        # print(grad_output)
        # print(type(grad_output))
        # print(1, grad_output.shape)
        if not hasattr(module, 'grad_tracks'):
            module.grad_tracks = []
        if not torch.is_tensor(grad_output):
            grad_output = [(out.abs()+1).detach().to(offload_device) for out in grad_output]
        else:
            grad_output = [(grad_output.abs()+1).detach().to(offload_device)]

        module.grad_tracks.append(grad_output)

    return track_grad_hook

track_grad_hook_to_cpu = generate_track_grad_hook('cpu')
track_grad_hook_to_cuda = generate_track_grad_hook('cuda')



def generate_track_input_hook(offload_device='cpu'):
    def track_input_hook(module, inputs, outputs):
        if not hasattr(module, 'input_tracks'):
            module.input_tracks = []
        if not torch.is_tensor(inputs):
            inputs = [inp.to(offload_device) for inp in inputs]
        else:
            inputs = [inputs.to(offload_device)]
        module.input_tracks.append(inputs)
    return track_input_hook

track_input_hook_to_cpu = generate_track_input_hook('cpu')
track_input_hook_to_cuda = generate_track_input_hook('cuda')


def save_input_output_to_disk_hook(module, inputs, outputs):

    assert not torch.is_tensor(inputs) # should be a list
    assert torch.is_tensor(outputs)

    if not torch.is_tensor(inputs):
        inputs = inputs[0]
    if not torch.is_tensor(outputs):
        outputs = outputs[0]

    name = module.name
    if not hasattr(module,"batch_count"):
        setattr(module,"batch_count", 0)
    else:
        module.batch_count+=1

    b_id =  module.batch_count

    save_dir =  os.path.join("hook_saves", name)
    os.makedirs(save_dir, exist_ok=True)
    input_save_name = os.path.join(save_dir, f'{b_id}_input.pth')
    out_save_name = os.path.join(save_dir, f'{b_id}_output.pth')

    # if os.path.exists(input_save_name):
    #     input_batches = torch.load(input_save_name,map_location="cpu")
    #     input_batches = torch.cat([input_batches, inputs.to("cpu")], dim=0)
    #     torch.save( input_batches, input_save_name )
    # else:
    torch.save(inputs.to("cpu"), input_save_name )

    # if os.path.exists(out_save_name):
    #     output_batches = torch.load(out_save_name,map_location="cpu")
    #     output_batches = torch.cat([output_batches, outputs.to("cpu")], dim=0)
    #     torch.save( output_batches, out_save_name)
    # else:
    torch.save(outputs.to("cpu"), out_save_name)



# the quantizer is designed for diffusion models with different timesteps quantized separately
class SdSeparateQuantizer(BaseQuantizer):
    def __init__(self, quant_hub_layer, wbit=Precision.FP16, abit=Precision.FP16, 
                 w_qscheme=torch.per_channel_affine, a_qscheme=torch.per_tensor_affine,offload='cpu', device='cuda', num_steps=None,
                 args=None, cali_data=None, model = None, name=None, relax_abit=10, relax_interval=(0.8,1)) -> None:
        super().__init__()
        self.wbit = wbit
        self.abit = abit
        self.quant_hub_layer = quant_hub_layer
        self.w_qscheme = w_qscheme
        self.a_qscheme = a_qscheme
        self.offload = offload
        self.device = device
        self.a_observers = []  # each t has its own quantizer
        self.num_steps = num_steps # total sampling steps
        self.curr_step = 0 # current sampling step, e.g., 0-49
        self.a_scales = [] # step to scale list
        self.a_zero_points = [] # zero point list
        self.gs = 1 # group size of timestep, 1 means using seperate quantizer for each timestep
        self.status = None
        self.finetune_step = -1

        # params for BRECQ
        self.alpha = None
        self.soft_targets = False
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.args = args
        self.cali_data = cali_data
        self.model = model
        self.quant_w = None
        self.name = name
        self.stored_activations = []

        self.relax_abit = relax_abit # bitwidth used for relaxation

        # self.relax_t = [] # no relax t 
        self.relax_t = list(range(int(num_steps*relax_interval[0]), int(num_steps*relax_interval[1])+1)) 
        self.recon = False


    
    
    def add_hook(self):
        if self.abit not in [Precision.FP16, Precision.FP32]:
            for t in range(0, (self.num_steps-1)//self.gs + 1):
                if t in self.relax_t:
                    abit = self.relax_abit
                else:
                    abit = self.abit
                # abit = self.step2bit[t]
                if self.a_qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    a_observer = PerChannelMinMaxObserver(
                        qscheme=self.a_qscheme,
                        quant_min=0,
                        quant_max=2 ** PRECISION_TO_BIT[abit] - 1
                    )
                elif self.a_qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                    a_observer = MovingAverageMinMaxObserver(
                        qscheme=self.a_qscheme,
                        quant_min=0,
                        quant_max=2 ** PRECISION_TO_BIT[abit] - 1
                    )

                self.a_observers.append(a_observer)
    

  # observe activation
    def observe(self, x):

        self.a_observers[self.curr_step//self.gs] = self.a_observers[self.curr_step//self.gs].to(x.device)
        self.a_observers[self.curr_step//self.gs](x)

    def set_curr_step(self, step):
        self.curr_step = step

    def set_finetune_step(self, step):
        self.finetune_step = step

    def set_recon_mark(self,state):
        self.recon = state

    # brecq @torch.no_grad()
    def quantize(self):
        # quantize weight
        W = self.quant_hub_layer.core.weight.to(self.device)
        if self.wbit not in [Precision.FP16, Precision.FP32]:
            logging.debug('quantzing linear weight')

            if not self.recon:
                # ori
                if self.w_qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    observer = PerChannelMinMaxObserver(
                        qscheme=self.w_qscheme,
                        quant_min=0,
                        quant_max=2 ** PRECISION_TO_BIT[self.wbit] - 1
                    )
                elif self.w_qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
                    observer = HistogramObserver(
                        qscheme=self.w_qscheme,
                        quant_min=0,
                        quant_max=2 ** PRECISION_TO_BIT[self.wbit] - 1
                    )
                observer = observer.to(W.device)
                observer(W)
                W = W.to(self.offload)
                scale, zero_point = observer.calculate_qparams()
                self.w_scale = scale.to(self.offload)
                self.w_zero_point = zero_point.to(self.offload)
                torch.cuda.empty_cache()
            else:
                # brecq
                W = W.to(self.offload)
                if self.w_qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    w_channel_wise = True
                else:
                    w_channel_wise = False
                scale, zero_point = self.init_quantization_scale(W, nbits=self.wbit, channel_wise=w_channel_wise)
                self.w_scale = scale.to(self.offload)
                self.w_zero_point = zero_point.to(self.offload)
                torch.cuda.empty_cache()
                self.init_alpha(w=W.clone())


        # quantize activation
        for a_observer in self.a_observers:
            if self.abit not in [Precision.FP16, Precision.FP32] and a_observer != None:
                #part quant
                try:
                    scale, zero_point = a_observer.calculate_qparams()
                except Exception:
                    # 32bit will overflow
                    a_scale, a_zero_point =0,0 #won't be used 
    
                a_scale = scale.to(self.offload)
                a_zero_point = zero_point.to(self.offload)
                torch.cuda.empty_cache()
                self.a_scales.append(a_scale)
                self.a_zero_points.append(a_zero_point)

        # delete layer inputs and outputs obtained by hook
        try:
            del self.quant_hub_layer.core.input_output_tracks
        except Exception:
            pass

    
    def forward(self, x):
        origin_dtype = x.dtype
        if self.abit == Precision.FP16:
            x = x.half()
        elif self.abit == Precision.FP32 and origin_dtype == torch.float32:
            x = x.float()
        elif self.status == "use_fp32":
            x = x.float()
        else:
            if self.curr_step == self.finetune_step:
                self.a_observers[self.curr_step//self.gs](x)   

            a_scale = self.a_scales[self.curr_step//self.gs]
            a_zero_point = self.a_zero_points[self.curr_step//self.gs]
            if self.curr_step in self.relax_t:
                abit = self.relax_abit
            else:
                abit = self.abit

            # skip when abit==32
            if abit == Precision.FP32:
                x = x.float()
            else:   
                if self.a_qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    x = torch.fake_quantize_per_channel_affine(
                        x,
                        a_scale.to(x.device),
                        a_zero_point.to(x.device),
                        0, 0, 2 ** PRECISION_TO_BIT[abit] - 1
                    ).to(x)
                else:
                    x = torch.fake_quantize_per_tensor_affine(
                        x,
                        a_scale.to(x.device),
                        a_zero_point.to(x.device),
                        0, 2 ** PRECISION_TO_BIT[abit] - 1
                    ).to(x)


        if self.wbit == Precision.FP16:
            w = self.quant_hub_layer.core.weight.half().to(x)
        elif self.wbit == Precision.FP32:
            w = self.quant_hub_layer.core.weight.float()
        elif self.status == "use_fp32":
            w = self.quant_hub_layer.core.weight.float()
        else:
            
            if not self.recon:
                if self.w_qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
                    w = torch.fake_quantize_per_channel_affine(
                        self.quant_hub_layer.core.weight,
                        self.w_scale.to(x.device),
                        self.w_zero_point.to(x.device),
                        0, 0, 2 ** PRECISION_TO_BIT[self.wbit] - 1
                    ).to(x)
                else:
                    w = torch.fake_quantize_per_tensor_affine(
                        self.quant_hub_layer.core.weight,
                        self.w_scale.to(x.device),
                        self.w_zero_point.to(x.device),
                        0, 2 ** PRECISION_TO_BIT[self.wbit] - 1
                    ).to(x)
            else:
                # BRECQ
                if self.soft_targets:
                    self.w_scale = self.w_scale.to(x)
                    self.w_zero_point = self.w_zero_point.to(x)
                    w_floor = torch.floor( self.quant_hub_layer.core.weight / self.w_scale)
                    w_int = w_floor + self.get_soft_targets()
                    w_quant = torch.clamp(w_int + self.w_zero_point, 0, 2 ** PRECISION_TO_BIT[self.wbit] - 1)
                    w = (w_quant - self.w_zero_point) * self.w_scale
                else:
                    if not hasattr(self,"quant_w") or self.quant_w is None:
                        w_scale = self.w_scale.to(x)
                        w_zero_point = self.w_zero_point.to(x)
                        w_floor = torch.floor( self.quant_hub_layer.core.weight.to(x) / w_scale)
                        w_int = w_floor + (self.alpha.to(self.device) >= 0).float()
                        w_quant = torch.clamp(w_int + w_zero_point, 0, 2 ** PRECISION_TO_BIT[self.wbit] - 1)
                        w = (w_quant - w_zero_point) * w_scale
                        self.quant_w = w
                        self.quant_w = self.quant_w.to(self.offload)
                    else:
                        # reduce memory
                        # self.quant_w = self.quant_w.to(self.offload)
                        # w = self.quant_w.to(x)
                        
                        self.quant_w = self.quant_w.to(x)
                        w = self.quant_w

        bias = None if self.quant_hub_layer.core.bias is None else self.quant_hub_layer.core.bias.to(x)
        if isinstance(self.quant_hub_layer.core, torch.nn.Conv2d):
            return F.conv2d(x, w, bias, self.quant_hub_layer.core.stride, self.quant_hub_layer.core.padding, self.quant_hub_layer.core.dilation, self.quant_hub_layer.core.groups).to(origin_dtype)
        elif isinstance(self.quant_hub_layer.core, torch.nn.Linear):
            return F.linear(x, w, bias).to(origin_dtype)
        else:
            raise NotImplementedError
        

    def init_quantization_scale(self, x, nbits, channel_wise = False):
        x = x.to("cuda")
        best_delta, best_zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            best_delta = x_max.clone()
            best_zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                best_delta[c], best_zero_point[c] = self.init_quantization_scale(x_clone[c], nbits,channel_wise=False)
            if len(x.shape) == 4:
                best_delta = best_delta.view(-1, 1, 1, 1)
                best_zero_point = best_zero_point.view(-1, 1, 1, 1)
            else:
                best_delta = best_delta.view(-1, 1)
                best_zero_point = best_zero_point.view(-1, 1)
        else:
            # MSE
            x_max = x.max()
            x_min = x.min()
            best_score = 1e+10
            for i in range(1):
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                # x_q = self.quantize(x, new_max, new_min)
                delta = (new_max - new_min) / (2 ** nbits - 1)
                zero_point = (-new_min / delta).round()
                # print(type(x), type(delta), type(zero_point))
                x_int =  torch.round(x/delta) + zero_point
                x_int = torch.clamp(x_int, 0, 2 ** PRECISION_TO_BIT[nbits] - 1)
                x_q = (x_int - zero_point)*delta
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    best_delta = (new_max - new_min) / (2 ** nbits - 1)
                    best_zero_point = (- new_min / best_delta).round()
       
        return best_delta, best_zero_point
    

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha.to(self.device)) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, w):

        w = w.cuda()
        self.w_scale = self.w_scale.cuda()
        w_floor = torch.floor(w / self.w_scale)
        print('Init alpha to be FP32')
        rest = (w / self.w_scale) - w_floor  # rest of rounding [0, 1)
        alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
        alpha = alpha.to(self.offload)
        self.alpha = torch.nn.Parameter(alpha)
        self.w_scale = self.w_scale.to(self.offload)


    def to(self, device):
        if hasattr(self, 'scale'):
            self.scale.to(device)
        if hasattr(self, 'zero_point'):
            self.zero_point.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def gpu(self):
        return self.to('cuda')

    def cuda(self):
        return self.to('cuda')
    
    def set_status(self, status):
        self.status = status
