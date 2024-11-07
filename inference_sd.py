'''
individually perform inference with the quantized ckpt
python inference_sd.py --model-path <path-to-ckpt> \
--save-path <save-path-for-imgs> \
--gpu_id 0 \
--batch-size 8 \
--path-to-test-text <path-to-test-prompts>
'''


from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import os
from PIL import Image
import numpy as np
import torch
from pytorch_lightning import seed_everything
from quantization_tools.utils.utils import find_layers
from quantization_tools.quantization.layers import LinearQuantHub, Conv2dQuantHub
import argparse
from quantization_tools.quantization.layers import MyStableDiffusionPipeline, MyStableDiffusionXLPipeline
import random



parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="/DATA/DATANAS1/tangsa/PCR_DiffBench/ckpt/quant-SD-w8a8-id1-re-8-8-8-8.ckpt")
parser.add_argument('--save-path', type=str, default="outputs/imgs")
parser.add_argument('--seed', type=int, default=999)
parser.add_argument('--gpu_id', type=str, default="0")
parser.add_argument('--interaction', action="store_true")
parser.add_argument('--SDXL', action="store_true", help="use stale diffusion XL")
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--path-to-test-text', type=str)
parser.add_argument('--num-generate', type=int, default=5000)
parser.add_argument('--coco', action="store_true",)
args = parser.parse_args()

sampling_steps = 50 

if args.SDXL:
    width = 768
    height = 768
    real_sample_steps = sampling_steps
    eff_mem = True

else:
    width = 512
    height = 512 
    real_sample_steps = sampling_steps+1 # a warmup step for pdnm 
    eff_mem = False

all_quant_layers = {}

def step_start_callback(step: int, timestep: int):

    for name, layer in all_quant_layers.items(): 
        for quantizer in layer.quantizer:
            # set current step for next denosing
            quantizer.set_curr_step(step)

def process_to_name(origin):
    oo = ''
    for o in origin:
        if o.lower() !=' ':
            oo += o
        elif o == ' ':
            oo += '_'
    return oo

def inference_interaction():
    seed_everything(args.seed)
    seeds = np.random.randint(low=100, high=50000, size=(30000,))
    global all_quant_layers
    # device = f"cuda:{args.gpu_id}"
    os.makedirs(args.save_path, exist_ok= True)
    if args.SDXL:
        base_pipe = MyStableDiffusionXLPipeline.from_pretrained("/DATA/DATANAS1/tangsa/huggingface/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/f898a3e026e802f68796b95e9702464bac78d76f", safety_checker=None, local_files_only = True, use_safetensors=True)
        # base_pipe.to("cuda")
    else:
        base_pipe = MyStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, local_files_only = True )

    print("load quantized checkpoint: ", args.model_path)
    quant_pipe = torch.load(args.model_path, map_location="cpu")
    layers_linear = find_layers(quant_pipe.unet, (LinearQuantHub, ))
    layers_conv = find_layers(quant_pipe.unet, (Conv2dQuantHub, ))
    all_quant_layers = {**layers_linear, **layers_conv}
    c = 0
    while True:
        text = input("Input your prompt: ")
        text = text.strip()
        if text == "exit":
            exit(0)
        seed_everything(seeds[c])
        base_pipe.to('cuda')
        base_out = base_pipe(text, output_type="np", width=width, height=height).images[0]
        base_pipe.to('cpu')
        seed_everything(seeds[c])
        quant_pipe.to("cuda")
        quant_out = quant_pipe(text, callback_on_start=step_start_callback, output_type="np", width=width, height=height).images[0]
        quant_pipe.to("cpu")
        final_out = np.concatenate([base_out, quant_out], axis=1)
        final_out = (final_out*255).astype("uint8")
        final_out = Image.fromarray(final_out)
        final_out.save(os.path.join(args.save_path,f"{c}_{process_to_name(text)}.png"))
        print(f"image is saved in {args.save_path}.")
        c+=1


def inference():
    seed_everything(args.seed)
    if not args.coco:
        txt_test = open(args.path_to_test_text, 'r').readlines()[0:args.num_generate]
        # txt_test = random.sample(list(txt_test), args.num_generate)
        txt_test = [d.strip() for d in txt_test]
    else:
        txt_test = open(args.path_to_test_text, 'r').readlines()
        txt_test = random.sample(list(txt_test), args.num_generate)
        txt_test = [d.strip() for d in txt_test]


    global all_quant_layers
    os.makedirs(args.save_path, exist_ok= True)

    print("load quantized checkpoint: ", args.model_path)
    quant_pipe = torch.load(args.model_path, map_location="cpu").to(f"cuda:{args.gpu_id}")

    # # base
    # if args.SDXL:S
    #     quant_pipe = MyStableDiffusionXLPipeline.from_pretrained("/DATA/DATANAS1/tangsa/huggingface/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/f898a3e026e802f68796b95e9702464bac78d76f", safety_checker=None, local_files_only = True, use_safetensors=True)
    # else:
    #     quant_pipe = MyStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, local_files_only = True ).to("cuda")
    
    if args.SDXL:
        quant_pipe.enable_model_cpu_offload()
    layers_linear = find_layers(quant_pipe.unet, (LinearQuantHub, ))
    layers_conv = find_layers(quant_pipe.unet, (Conv2dQuantHub, ))
    all_quant_layers = {**layers_linear, **layers_conv}
    with torch.no_grad():
        start = 0
        bs = args.batch_size
        while True:
            print("generate test num:", start)
            batch = txt_test[start:start + bs]          
            imgs = quant_pipe(batch, callback_on_start=step_start_callback, width=width, height=height).images
            for i, img in enumerate(imgs):
                try:
                    img.save(os.path.join(args.save_path, f"{str(i+1+start)}_{process_to_name(batch[i].strip())}.png"))
                except Exception:
                    ...
            start += bs
            if start >= len(txt_test):
                break
    print("finish test")
 


if __name__ == "__main__":
    if args.interaction:
        inference_interaction()
    else:
        inference()
