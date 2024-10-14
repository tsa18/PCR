import torch
from tqdm import tqdm
from . import QuantizedModule
from . import Precision
from .loss import BrecqLoss, lp_loss
from .quantizer import track_input_output_hook_to_cpu, track_grad_hook_to_cpu 
from quantization_tools.quantization.layers import ResnetBlock2DQuantHub, BasicTransformerBlockQuantHub
import torch.nn.functional as F
import time
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub
from . import Precision


def find_layers(module, layers, name=''):
    if isinstance(layers, list):
        layers = tuple(layers)
    if isinstance(module, layers):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def set_quantize_status(all_quant_layers, state:str):
    for name, layer in all_quant_layers.items(): 
        layer.status = state


def quantize_model_till(model, layer, all_quant_layers):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    set_quantize_status(all_quant_layers, state="just_core")
    for name, module in model.named_modules():
        if isinstance(module, (LinearQuantHub, Conv2dQuantHub )):
            module.status = 'quantized'
        if module == layer:
            break


# block reconstruction for brecq
def block_reconstruction(model, all_quant_layers, block, cali_data, batch_size: int = 32, iters: int = 2000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, prepare_data = False,
                        data_num = 10, bs=4, num_inference_steps=50, guidance_scale=7.5, ptq4dm_data = None, acc_n = 1,
                        width=512, height=512, eff_mem = False, SDXL = False,):
        """
        Block reconstruction to optimize the output from each layer.

        :param model: QuantModel
        :param block: QuantBlock that needs to be optimized
        :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
        :param batch_size: mini-batch size for reconstruction
        :param iters: optimization iterations for reconstruction,
        :param weight: the weight of rounding regularization term
        :param opt_mode: optimization mode
        :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
        :param include_act_func: optimize the output after activation function
        :param b_range: temperature range
        :param warmup: proportion of iterations that no scheduling for temperature
        :param act_quant: use activation quantization or not.
        :param lr: learning rate for act delta learning
        :param p: L_p norm minimization
        :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
        """
        # model.set_quant_state(False, False)
        # layer.set_quant_state(True, act_quant)

        # if not include_act_func:
        #     org_act_func = layer.activation_function
        #     layer.activation_function = StraightThrough()

        print("############################ start block reconstruction ############################", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        quantizers = []

        # only one quantizer in the list
        #find all layers in block
        layers_in_block = find_layers(block, (QuantizedModule,))
        for name, layer in layers_in_block.items():
            quantizers.append(layer.quantizer[0])


        if not act_quant:
            for quantizer in quantizers:
                quantizer.soft_targets = True
            # Set up optimizer
            opt_params = []
            for quantizer in quantizers:
                opt_params.append(quantizer.alpha)
            optimizer = torch.optim.Adam(opt_params)
            scheduler = None
            
            #disable activation quant
            for quantizer in quantizers:
                quantizer.tmp_abit = quantizer.abit
                quantizer.abit =  Precision.FP32 
        else:
            # learn scaling factor for activation
            opt_params = []
            for quantizer in quantizers:
                opt_params.append(quantizer.a_scale)
            optimizer = torch.optim.Adam(opt_params, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

        loss_mode = 'none' if act_quant else 'relaxation'
        rec_loss = opt_mode

        loss_func =  BlockLossFunction(round_loss=loss_mode, weight=weight,
                                max_count=iters, rec_loss=rec_loss, b_range=b_range,
                                decay_start=0, warmup=warmup, p=p)


        # Save data before optimizing the rounding
        data_num = data_num
        bs = bs
        block.record_inout = True
        print("start prepare inputs and outputs")
        set_quantize_status(all_quant_layers, state="just_core")
        if eff_mem:
            model.enable_sequential_cpu_offload()
        with torch.no_grad():
            if opt_mode != 'mse':
                model.set_record_inputs(True)
                model.init_record_inputs()
            for i in range(0, data_num, bs):
                if cali_data is not None: # text-to-image
                    model(cali_data[i:i+bs], output_type = "latent", width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
                else:
                    model(batch_size= bs, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) #unconditional
                torch.cuda.empty_cache()
            if opt_mode != 'mse':
                model.set_record_inputs(False)
        print("finish prepare inputs and outputs")
        if eff_mem:
            remove_sequential_cpu_load(model)
            model.to('cpu')
        torch.cuda.empty_cache()

        block.record_inout = False
        set_quantize_status(all_quant_layers, state="quantized")

        cached_inps=[]
        cached_outs=[]

        input_num = len(block.inputs[0])
        for _ in range(input_num):
            cached_inps.append([])
        for input in block.inputs:
            for k in range(len(input)):
                cached_inps[k].append(input[k])
        for output in block.outputs:
            cached_outs.append(output)
        
        # save grad
        if opt_mode != 'mse':
            print("start prepare grad outputs")
            set_quantize_status(all_quant_layers, state="just_core")      
            hook = block.register_backward_hook(track_grad_hook_to_cpu)
            for input in tqdm(model.inputs, desc="calc L grad"):
                model.unet.zero_grad()
                input = [inp.to(quantizer.device) for inp in input]
                with torch.no_grad():
                    # set_quantize_status(all_quant_layers, state="quantized")
                    last_layer_key = list(layers_in_block.keys())[-1]
                    last_layer = layers_in_block[last_layer_key]
                    quantize_model_till(model.unet, layer=last_layer, all_quant_layers=all_quant_layers)
                    out_q = model.forward_unet(*input)
                with torch.enable_grad():
                    set_quantize_status(all_quant_layers, state="just_core")   
                    out_fp = model.forward_unet(*input)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()

            hook.remove()
            set_quantize_status(all_quant_layers, state="quantized")
            print("finish prepare grad outputs")
            cached_grads = []
            for grad_out in block.grad_tracks:
                cached_grads.append(grad_out[0])
        else:
            cached_grads = None

        all_num = 0
        for out in cached_outs:
            all_num += out.shape[0]
        real_bs = cached_outs[0].shape[0]

        # only reserve the block in gpu
        # model.to('cpu')
        block.to(quantizer.device)
      

        for i in tqdm(range(iters)):
            # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
            idx = torch.randperm(all_num)[:batch_size]

            # cur_inp = cached_inps[idx].to(quantizer.device)
            cur_inp = []
            cur_out = []

            for i in idx:
                cur_out.append(cached_outs[i//real_bs][i%real_bs])
            cur_out = torch.stack(cur_out,dim=0).to(quantizer.device)

            for k in range(len(cached_inps)):
                if torch.is_tensor(cached_inps[k][0]):
                    tmp_in = []
                    for i in idx:
                        tmp_in.append(cached_inps[k][i//real_bs][i%real_bs])
                    cur_inp.append(torch.stack(tmp_in, dim=0).to(quantizer.device))
                else:
                    cur_inp.append(cached_inps[k][0])
        
            # cur_out = cached_outs[idx].to(quantizer.device)
            # cur_grad = cached_grads[idx] if opt_mode != 'mse' else None
            if opt_mode != 'mse':
                cur_grad = []
                for i in idx:
                    cur_grad.append(cached_grads[i//real_bs][i%real_bs])  
                cur_grad = torch.stack(cur_grad, dim=0).to(quantizer.device)
            else:
                cur_grad = None

            # optimizer.zero_grad()
            out_quant = block(*cur_inp)

            # err = loss_func(out_quant, cur_out, quantizers, cur_grad)
            err = loss_func(out_quant, cur_out, quantizers, cur_grad) / acc_n
            err.backward(retain_graph=True)
            if (i+1) % acc_n ==0:
                optimizer.step()
                optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        torch.cuda.empty_cache()

        # Finish optimization, use hard rounding.
        for quantizer in quantizers:
            quantizer.soft_targets = False

        print("############################ finish block reconstruction ############################",  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        try:
            del block.inputs
            del block.outputs
            del block.grad_tracks

        except Exception:
            ...

        # enable activation quant
        for quantizer in quantizers:
            if hasattr(quantizer,"tmp_abit"):
                quantizer.abit = quantizer.tmp_abit 
    
        # # Reset original activation function
        # if not include_act_func:
        #     layer.activation_function = org_act_func

        # only reserve the block in gpu
        # model.to(quantizer.device)
        if eff_mem:
            block.to('cpu')
            model.to('cpu')


class BlockLossFunction:
    def __init__(self,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        # self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, quantizers, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for quantizer in quantizers:
                round_vals = quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


def layer_reconstruction(model, all_quant_layers, layer, cali_data, batch_size: int = 32, iters: int = 2000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, prepare_data = False,
                        data_num = 10, bs=4, num_inference_steps=50, guidance_scale=7.5, acc_n = 1,
                        width=512, height=512, eff_mem= False, SDXL = False):
        """
        Block reconstruction to optimize the output from each layer.

        :param model: QuantModel
        :param layer: QuantModule that needs to be optimized
        :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
        :param batch_size: mini-batch size for reconstruction
        :param iters: optimization iterations for reconstruction,
        :param weight: the weight of rounding regularization term
        :param opt_mode: optimization mode
        :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
        :param include_act_func: optimize the output after activation function
        :param b_range: temperature range
        :param warmup: proportion of iterations that no scheduling for temperature
        :param act_quant: use activation quantization or not.
        :param lr: learning rate for act delta learning
        :param p: L_p norm minimization
        :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
        """
        # model.set_quant_state(False, False)
        # layer.set_quant_state(True, act_quant)

        # if not include_act_func:
        #     org_act_func = layer.activation_function
        #     layer.activation_function = StraightThrough()

        print("############################ start layer reconstruction ############################",  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
        quantizer = layer.quantizer[0]

        # skip
        if quantizer.wbit in [Precision.FP16, Precision.FP32]:
            print(f"skip the layer {layer.name}")
            return

        if not act_quant:
            quantizer.soft_targets = True
            # Set up optimizer
            opt_params = [quantizer.alpha]
            optimizer = torch.optim.Adam(opt_params)
            scheduler = None

            #disable activation quant
            quantizer.tmp_abit = quantizer.abit
            quantizer.abit =  Precision.FP32 

        else:
            # learn scaling factor for activation
            opt_params = [quantizer.a_scale]
            optimizer = torch.optim.Adam(opt_params, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

        loss_mode = 'none' if act_quant else 'relaxation'
        rec_loss = opt_mode

        loss_func =  BrecqLoss(round_loss=loss_mode, weight=weight,
                                max_count=iters, rec_loss=rec_loss, b_range=b_range,
                                decay_start=0, warmup=warmup, p=p)


        # Save data before optimizing the rounding
        data_num = data_num
        bs = bs

        print("start prepare inputs and outputs")
        set_quantize_status(all_quant_layers, state="just_core")
        if eff_mem:
            model.enable_sequential_cpu_offload()
        with torch.no_grad():
            if opt_mode != 'mse':
                model.set_record_inputs(True)
                model.init_record_inputs()
            hook = layer.core.register_forward_hook(track_input_output_hook_to_cpu)
            for i in range(0, data_num, bs):
                if cali_data is not None:
                    model(cali_data[i:i+bs], output_type = "latent", width=width, height=height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
                else:
                    model(batch_size= bs, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
                torch.cuda.empty_cache()
            if opt_mode != 'mse':
                model.set_record_inputs(False)
        hook.remove()
        set_quantize_status(all_quant_layers, state="quantized")
        print("finish prepare inputs and outputs")
        if eff_mem:
            remove_sequential_cpu_load(model)
            model.to('cpu')
        torch.cuda.empty_cache()

        cached_inps=[]
        cached_outs=[]
        for inputs, outputs in layer.core.input_output_tracks:
            cached_inps.append(inputs[0])
            cached_outs.append(outputs[0])

        # save grad
        if opt_mode != 'mse':
            print("start prepare grad outputs")
            set_quantize_status(all_quant_layers, state="just_core")      
            hook = layer.core.register_backward_hook(track_grad_hook_to_cpu)
            for input in tqdm(model.inputs, desc="calc L grad"):
                model.unet.zero_grad()
                input = [inp.to(quantizer.device) for inp in input]
                with torch.no_grad():
                    # set_quantize_status(all_quant_layers, state="quantized")
                    quantize_model_till(model.unet, layer=layer, all_quant_layers=all_quant_layers)
                    out_q = model.forward_unet(*input)
                with torch.enable_grad():
                    set_quantize_status(all_quant_layers, state="just_core")   
                    out_fp = model.forward_unet(*input)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()

            hook.remove()
            set_quantize_status(all_quant_layers, state="quantized")
            print("finish prepare grad outputs")
            cached_grads = []
            for grad_out in layer.core.grad_tracks:
                cached_grads.append(grad_out[0])
        else:
            cached_grads = None
        

        all_num = 0
        for out in cached_outs:
            all_num += out.shape[0]
        real_bs = cached_outs[0].shape[0]

        # only reserve the block in gpu
        # model.to('cpu')
        layer.to(quantizer.device)

        for i in tqdm(range(iters)):
            cur_inp = []
            cur_out = []
            idx = torch.randperm(all_num)[:batch_size]
            # idx = torch.randperm(cached_inps.size(0))[:batch_size]
            for i in idx:
                cur_inp.append(cached_inps[i//real_bs][i%real_bs])  
                cur_out.append(cached_outs[i//real_bs][i%real_bs])
            cur_inp = torch.stack(cur_inp,dim=0).to(quantizer.device)
            cur_out = torch.stack(cur_out,dim=0).to(quantizer.device)

            if opt_mode != 'mse':
                cur_grad = []
                for i in idx:
                    cur_grad.append(cached_grads[i//real_bs][i%real_bs])  
                cur_grad = torch.stack(cur_grad, dim=0).to(quantizer.device)
            else:
                cur_grad = None
            # cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

            # optimizer.zero_grad()
            # out_quant = layer(cur_inp)
            out_quant = quantizer.forward(cur_inp)

            err = loss_func(out_quant, cur_out, quantizer.get_soft_targets(), cur_grad) /acc_n
            err.backward(retain_graph=True)

            if (i+1) % acc_n ==0:
                optimizer.step()
                optimizer.zero_grad()

            # optimizer.step()
            if scheduler:
                scheduler.step()

        torch.cuda.empty_cache()

        # Finish optimization, use hard rounding.
        quantizer.soft_targets = False

        print("############################ finish layer reconstruction ############################",  time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))

        try:
            del layer.core.input_output_tracks
            del layer.core.grad_tracks
        except Exception:
            pass
        # enable activation quant
        if hasattr(quantizer,"tmp_abit"):
            quantizer.abit = quantizer.tmp_abit 

        # # Reset original activation function
        # if not include_act_func:
        #     layer.activation_function = org_act_func
        # model.to(quantizer.device)
        if eff_mem:
            layer.to('cpu')
            model.to('cpu')
        # reporter = MemReporter(layer)
        # reporter.report()
    

from accelerate import hooks

def remove_sequential_cpu_load(pipeline):
  for module in [pipeline.unet, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.vae]:
        hooks.remove_hook_from_submodules(module)
