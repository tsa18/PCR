"""
perform PTQ for SD and do inference with the quantized model

CUDA_VISIBLE_DEVICES=5 python benchmark/eval-sd-test.py \
    --linear-a-bit 8 \
    --linear-w-bit 8 \
    --conv-a-bit 8 \
    --conv-w-bit 8 \
    --num-calibrate 200 \
    --cali-batch-size 4 \
    --num-generate 5000  \
    --model-path CompVis/stable-diffusion-v1-4 # the path of your pretrained SD model \
    --path-to-test-text prompts/coco_val2017.txt # test prompts \
    --path-to-cali-text prompts/coco_train2017.txt # calibration prompts \
    --exp-name SD_w8a8_coco # exp name \
    --method Separate  --relax-first-last-layer --recon \
    --prog # progressive calibration \
    --relax_interval_s 0.8  --relax_interval_e 1.0 # set relaxation intervals
"""

from pytorch_lightning import seed_everything
import os, sys
sys.path.append(os.getcwd())
import torch
from tqdm import tqdm
from quantization_tools.utils.utils import replace_module, find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub
from quantization_tools.quantization.layers import ResnetBlock2DQuantHub, BasicTransformerBlockQuantHub, MyStableDiffusionPipeline, MyStableDiffusionXLPipeline
from quantization_tools.quantization.quantizer import  SdSeparateQuantizer
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import DiffusionPipeline, StableDiffusionPipeline
# from diffusers import LCMScheduler
import argparse
import random
from quantization_tools.utils.evaluation import calculate_fid_given_paths, InceptionV3, calculate_clip_score_dir
import csv
from diffusers.models.unet_2d_blocks import UpBlock2D,  CrossAttnUpBlock2D
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.unet_2d_blocks import ResnetBlock2D
from diffusers import DPMSolverMultistepScheduler, UniPCMultistepScheduler
from PIL import Image
import shutil
import resource
from quantization_tools.quantization.quantizer_utils import block_reconstruction, layer_reconstruction
import time
from quantization_tools.quantization import Precision
from accelerate import hooks
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--linear-a-bit', type=int, default=16)
parser.add_argument('--linear-w-bit', type=int, default=4)
parser.add_argument('--conv-a-bit', type=int, default=16)
parser.add_argument('--conv-w-bit', type=int, default=4)
parser.add_argument('--num-calibrate', type=int, default=1)
parser.add_argument('--num-generate', type=int, default=5)
parser.add_argument('--model-path', type=str)
parser.add_argument('--path-to-cali-text', type=str)
parser.add_argument('--path-to-test-text', type=str)
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size for evaluation')
parser.add_argument('--cali-batch-size', type=int, default=8,
                    help='Batch size for calibration/inference')
parser.add_argument('--save-ckpt-dir', type=str, default="ckpt")
parser.add_argument('--use-ckpt', action='store_true')
parser.add_argument('--use-ckpt-path', type=str, default=None)
parser.add_argument('--exp-name', type=str, default="exp", help="the name of this exp")
parser.add_argument('--method', type=str, default="RTN", help="choose the quantization method, e.g, BRECQ")
parser.add_argument('--separate', action="store_true", help="separately quantize different time steps")
parser.add_argument('--recon', action="store_true", help="use the block-wise reconstruction")
parser.add_argument('--prog', action="store_true", help="progressive calibration")
parser.add_argument('--relax_interval_s', type=float, default=0.8,help="interval start for act relaxation")
parser.add_argument('--relax_interval_e', type=float, default=1.0, help="interval end for act relaxation")
parser.add_argument('--relax-first-last-layer', action="store_true", help="use higher bitwidth for the first and the last layer")
parser.add_argument('--SDXL', action="store_true", help="use stale diffusion XL")


methods ={
    # "RTN": [LinearRTNQuantizer, Conv2dRTNQuantizer],
    "Separate": [SdSeparateQuantizer, SdSeparateQuantizer],
}

# import resource
# # set the maximum memory to 350GB
resource.setrlimit(resource.RLIMIT_AS, (350*1024 * 1024 * 1024, -1))

args = parser.parse_args()


seed_everything(23)

all_quant_layers ={}
# set sampling steps, default of diffuser pipeline is 50
sampling_steps = 50 


if args.SDXL:
    width = 768
    height = 768
    real_sample_steps = sampling_steps
    eff_mem = True
    guidance_scale= 7.5

else:
    width = 512
    height = 512 
    real_sample_steps = sampling_steps+1 # a warmup step for pdnm 
    eff_mem = False
    guidance_scale= 7.5



def process_to_name(origin):
    oo = ''
    for o in origin:
        if o.lower() !=' ':
            oo += o
        elif o == ' ':
            oo += '_'
    return oo


def step_start_callback(step: int, timestep: int):
    if args.method == "Separate":
        for name, layer in all_quant_layers.items(): 
            for quantizer in layer.quantizer:
                # set current step for next denosing
                quantizer.set_curr_step(step)
    else:
        raise NotImplementedError


def set_no_quantized(all_quant_layers):
    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            quantizer.set_status("use_fp32")

def set_quantized(all_quant_layers):
    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            quantizer.set_status("quantized")


def reset_a_observers(all_quant_layers, step):
    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            if quantizer.abit in [Precision.FP16, Precision.FP32]:
                continue
            quantizer.a_observers[step-1].reset_min_max_vals()

def set_new_params(all_quant_layers, step):
    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            if quantizer.abit in [Precision.FP16, Precision.FP32]:
                continue
            scale, zero_point = quantizer.a_observers[step-1].calculate_qparams()
            quantizer.a_scales[step-1]= scale.to(quantizer.offload)
            quantizer.a_zero_points[step-1] = zero_point.to(quantizer.offload)


def set_finetune_step(all_quant_layers, step):
    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            quantizer.set_finetune_step(step-1)

## progressive calibration
def progressive_act_calib(pipeline, all_quant_layers, cali_texts):
 
    cali_texts = cali_texts[:]
    # finetune_steps = list(range(0, real_sample_steps))
    finetune_steps = list(range(1, real_sample_steps))

    bs = args.cali_batch_size 
    reset = True

    # calibrate step t
    for t in finetune_steps:
        print("finetuning",t)
        set_finetune_step(all_quant_layers, t)
        if reset:
            reset_a_observers(all_quant_layers, t)
        for start in range(0, len(cali_texts), bs):
            print("finetuning num:",start)
            batch = cali_texts[start:start + bs]         
            pipeline(batch, callback_on_start=step_start_callback, max_inference_steps = t, width=width, height=height, num_inference_steps=sampling_steps, guidance_scale=guidance_scale)
        set_new_params(all_quant_layers, t)

    #finish
    set_finetune_step(all_quant_layers, -1)
    
        
# Reconstruct model with BRECQ
def recon_model(model):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, (LinearQuantHub, Conv2dQuantHub)):
                print('Reconstruction for layer {}: {}'.format(name, module.name))
                if args.SDXL:
                    layer_reconstruction(pipeline, all_quant_layers, module, cali_data=txt_cali, bs=8, batch_size=8, acc_n=1, iters=2000, data_num=8,  width=width, height=height, eff_mem=eff_mem, SDXL = args.SDXL)
                else:
                    layer_reconstruction(pipeline, all_quant_layers, module, cali_data=txt_cali, bs=1, batch_size=32, acc_n=1, iters=2000, data_num=10, 
                                         width=width, height=height, eff_mem=eff_mem, SDXL = args.SDXL, num_inference_steps=sampling_steps, guidance_scale=guidance_scale)
                
                
            if isinstance(module, (ResnetBlock2DQuantHub, BasicTransformerBlockQuantHub)):
                print('Reconstruction for block {}: {}'.format(name, module.name))
                if args.SDXL:
                    block_reconstruction(pipeline, all_quant_layers, module, cali_data=txt_cali, bs=8, batch_size=8, acc_n=1, iters=2000, data_num=8, width=width, height=height, eff_mem=eff_mem, SDXL = args.SDXL)
                else:
                    block_reconstruction(pipeline, all_quant_layers, module, cali_data=txt_cali, bs=1, batch_size=32, acc_n=1, iters=2000, data_num=10, 
                                         width=width, height=height, eff_mem=eff_mem, SDXL = args.SDXL, num_inference_steps=sampling_steps, guidance_scale=guidance_scale)
            else:
                recon_model(module)


if __name__ == '__main__':
    
    # load calibration data
    txt_cali = open(args.path_to_cali_text, 'r').readlines()
    txt_cali = random.sample(list(txt_cali), args.num_calibrate)
    txt_cali = [d.strip() for d in txt_cali]

    if not args.use_ckpt:
        # load pretrained checkpoint
        if args.SDXL:
            print("sdxl")
            pipeline = MyStableDiffusionXLPipeline.from_pretrained("/DATA/DATANAS1/tangsa/huggingface/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/f898a3e026e802f68796b95e9702464bac78d76f", safety_checker=None, local_files_only = True, use_safetensors=True)
            pipeline.to("cuda")

        else:
            pipeline = MyStableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, local_files_only = True ).to('cuda')

        model = pipeline.unet
        # replace linear modules
        if args.linear_a_bit <= 8 or args.linear_w_bit <= 8:
            print('\n==== replace linear modules ====')
            replace_module(model, torch.nn.Linear, LinearQuantHub, display=True)
            layers_linear = find_layers(model, (LinearQuantHub, ))
            for name, layer in layers_linear.items(): #tqdm(layers_linear.items()):
                layer: LinearQuantHub
                kargs = {}
                if args.method == "Separate":
                    kargs.update({'num_steps': real_sample_steps,
                                  'relax_interval': (args.relax_interval_s, args.relax_interval_e)})

                layer.register_quantizer(methods[args.method][0](layer, args.linear_w_bit, args.linear_a_bit, **kargs))
                layer.prepare_hook()
        else:
            layers_linear = {}

        # replace conv modules
        if args.conv_a_bit <= 8 or args.conv_w_bit <= 8:
            print('\n==== replace conv2d modules ====')
            replace_module(model, torch.nn.Conv2d, Conv2dQuantHub, display=True)
            layers_conv = find_layers(model, (Conv2dQuantHub, ))
            for name, layer in layers_conv.items(): #tqdm(layers_conv.items()):
                layer: Conv2dQuantHub
                kargs = {}
                kargs.update({'name': name})
                if args.method == "Separate":
                    kargs.update({'num_steps': real_sample_steps,
                                  'relax_interval': (args.relax_interval_s, args.relax_interval_e)})
                if args.relax_first_last_layer and (name == "conv_in" or name == "conv_out"):
                    print(f"relax quantization for layer {name}")
                    layer.register_quantizer(methods[args.method][1](layer, Precision.FP16, Precision.FP16, **kargs))
                else:
                    layer.register_quantizer(methods[args.method][1](layer, args.conv_w_bit, args.conv_a_bit, **kargs))
                layer.prepare_hook()
        else:
            layers_conv = {}

        # replace blocks for block construction later
        replace_module(model, ResnetBlock2D, ResnetBlock2DQuantHub, display=True)
        replace_module(model, BasicTransformerBlock, BasicTransformerBlockQuantHub, display=True)

        all_quant_layers = {**layers_linear, **layers_conv}
        print(f"all layers num: {len(all_quant_layers)}")

        for name in list(all_quant_layers.keys()):
            layer = all_quant_layers[name]
            setattr(layer,'name',name)
    
        # calibration, iterate through some data
        origin_path = f"./outputs/outputs_{args.exp_name}/quantized{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}/origin"
        os.makedirs(origin_path, exist_ok=True)
        # save current experimental codes
        shutil.copy("benchmark/eval-sd-test.py", f"./outputs/outputs_{args.exp_name}/quantized{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}/")
        shutil.copy("quantization_tools/quantization/quantizer_utils.py", f"./outputs/outputs_{args.exp_name}/quantized{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}/")
        shutil.copy("quantization_tools/quantization/quantizer.py", f"./outputs/outputs_{args.exp_name}/quantized{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}/")


        # original calibration
        with torch.no_grad():
            start = 0
            cali_bs = args.cali_batch_size
            while True:
                print("start calibrate num:", start)
                cali_batch = txt_cali[start:start + cali_bs]
                imgs = pipeline(cali_batch, callback_on_start=step_start_callback, width=width, height=height, num_inference_steps=sampling_steps, guidance_scale=guidance_scale).images
                # imgs = pipeline(cali_batch, width=width, height=height, num_inference_steps=sampling_steps, guidance_scale=guidance_scale).images
                for i, img in enumerate(imgs):
                    try:
                        img.save(os.path.join(origin_path, f"{str(i+1+start)}_{process_to_name(cali_batch[i])}.png"))
                    except Exception:
                        ...
                start += cali_bs
                if start >= len(txt_cali):
                    break

        print("Calibration finished!")
        print('\n==== quantizing layers ====')

        # whether to use weight reconstruction as BRECQ
        if args.recon:
            for name, layer in all_quant_layers.items(): 
                for quantizer in layer.quantizer:
                    quantizer.set_recon_mark(True)
                    
        # quantize         
        for name, layer in tqdm(all_quant_layers.items()): 
            # prepare for brecq 
            if args.method == "BRECQ" or args.method == "Separate":
                for quantizer in layer.quantizer:
                    # if isinstance(quantizer,(LinearAdaRoundQuantizer,Conv2dAdaRoundQuantizer)):
                    if isinstance(quantizer,(SdSeparateQuantizer)):
                        quantizer.model = pipeline
                        quantizer.args = args
                        quantizer.cali_data = txt_cali

            layer.remove_hook()
            layer.quantize()
 
        # set status to be quantized
        for name, layer in tqdm(all_quant_layers.items()): 
            layer.set_default_quantizer(0)
            layer.to(layer.core.weight.device)
       
        pipeline.to('cuda')
        print("Quantization finished!")

        if all_quant_layers:
            if args.method == "Separate":
                ## block reconstuction
                if args.recon:
                    print("---------start reconstruction-----------", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
                    recon_model(pipeline.unet)
                    print("-----------finish reconstruction-----------", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
                    # save quantized ckpt
                    os.makedirs(args.save_ckpt_dir, exist_ok=True)
                    save_path = os.path.join(args.save_ckpt_dir,f"quant-{args.exp_name}-{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}.ckpt")
                    try:
                        torch.save(pipeline, save_path)
                        print("Quantized ckpt saved before prog!") 
                    except Exception as e:
                        pass
            
                    pipeline.to("cuda")
                    if eff_mem:
                        pipeline.enable_model_cpu_offload()

                if args.prog:
                    # Progressive calibration for act
                    print("---------start progressive-----------", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
                    progressive_act_calib(pipeline, all_quant_layers, cali_texts=txt_cali)
                    print("-----------finish progressive-----------", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))


        # save quantized ckpt
        os.makedirs(args.save_ckpt_dir, exist_ok=True)
        save_path = os.path.join(args.save_ckpt_dir,f"quant-{args.exp_name}-{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}.ckpt")
        try:
            torch.save(pipeline, save_path)
            print("Quantized ckpt saved!")
        except Exception as e:
            pass

    # load quantized ckpt
    else:
        assert args.use_ckpt_path is not None
        print(f"######### load ckpt from {args.use_ckpt_path} ###########")
        pipeline = torch.load(args.use_ckpt_path, map_location="cpu")
        if args.linear_a_bit <= 8 or args.linear_w_bit <= 8:
            layers_linear = find_layers(pipeline.unet, (LinearQuantHub, ))
        if args.conv_a_bit <= 8 or args.conv_w_bit <= 8:
            layers_conv = find_layers(pipeline.unet, (Conv2dQuantHub, ))
        all_quant_layers = {**layers_linear, **layers_conv}
        pipeline.to("cuda")  
        if eff_mem:
            pipeline.enable_model_cpu_offload()
        


    # test, generate data using the quantized model
    seed_everything(999)
    # pipeline.to("cuda")  

    txt_test = open(args.path_to_test_text, 'r').readlines()
    txt_test = random.sample(list(txt_test), args.num_generate)
    txt_test = [d.strip() for d in txt_test]

    res_path = f"./outputs/outputs_{args.exp_name}/quantized{args.linear_a_bit}-{args.linear_w_bit}-{args.conv_a_bit}-{args.conv_w_bit}/quantized"
    os.makedirs(res_path, exist_ok=True)
    # generate samples using quantized model
    with torch.no_grad():
        start = 0
        bs = args.cali_batch_size
        while True:
            print("generate test num:", start)
            batch = txt_test[start:start + bs]          
            imgs = pipeline(batch, callback_on_start=step_start_callback, width=width, height=height, num_inference_steps=sampling_steps, guidance_scale=guidance_scale).images
            for i, img in enumerate(imgs):
                try:
                    img.save(os.path.join(res_path, f"{str(i+1+start)}_{process_to_name(batch[i].strip())}.png"))
                except Exception:
                    ...
            start += bs
            if start >= len(txt_test):
                break
    print("Test finished!")


