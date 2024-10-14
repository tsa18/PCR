## Setup:
Execute the following commands to setup python environment:
```
create -n pcr python=3.10
conda activate pcr
pip install -r requirements.txt
```

## Run experiments
### Step 1. Quantize model and generate samples
#### For SD-v1.4 and 1.5
```
CUDA_VISIBLE_DEVICES=0 python benchmark/eval-sd-test.py \
    --linear-a-bit 8 \
    --linear-w-bit 8 \
    --conv-a-bit 8 \
    --conv-w-bit 8 \
    --num-calibrate 200 \
    --cali-batch-size 4 \
    --num-generate 5000  \
    --model-path CompVis/stable-diffusion-v1-4       # the path of your pretrained SD model \
    --path-to-test-text prompts/coco_val2017.txt     # test prompts \
    --path-to-cali-text prompts/coco_train2017.txt   # calibration prompts \
    --exp-name SD_w8a8_coco    # exp name \
    --method Separate  --relax-first-last-layer --recon \
    --prog # progressive calibration \
    --relax_interval_s 0.8  --relax_interval_e 1.0    # set relaxation intervals
```

#### For SDXL
```
CUDA_VISIBLE_DEVICES=0 python benchmark/eval-sd-test.py \
    --linear-a-bit 8 \
    --linear-w-bit 8 \
    --conv-a-bit 8 \
    --conv-w-bit 8 \
    --num-calibrate 200 \
    --cali-batch-size 4 \
    --num-generate 5000  \
    --model-path CompVis/stable-diffusion-v1-4       # the path of your pretrained SD model \
    --path-to-test-text prompts/coco_val2017.txt     # test prompts \
    --path-to-cali-text prompts/coco_train2017.txt   # calibration prompts \
    --exp-name SDXL_w8a8_coco    # exp name \
    --method Separate  --relax-first-last-layer --recon \
    --prog # progressive calibration \
    --relax_interval_s 0  --relax_interval_e 0.2   # set relaxation intervals \
    --SDXL 
```

### Step 2. Evaluate quantized models (Calculate FID-to-FP32)
#### Generate 5k samples using FP32 model
```
CUDA_VISIBLE_DEVICES=0 python benchmark/eval-sd-test.py \
    --linear-a-bit 32 \
    --linear-w-bit 32 \
    --conv-a-bit 32 \
    --conv-w-bit 32 \
    --num-calibrate 200 \
    --cali-batch-size 4 \
    --num-generate 5000  \
    --model-path CompVis/stable-diffusion-v1-4       # the path of your pretrained SD model \
    --path-to-test-text prompts/coco_val2017.txt     # test prompts \
    --path-to-cali-text prompts/coco_train2017.txt   # calibration prompts \
    --exp-name SD_FP32_coco    # exp name \
    --method Separate
```
#### Calculate FID bewteen FP32 samples and quantized samples
```
fidelity --fid --input1  <path-to-fp32-img-dir> --input2  <path-to-quant-img-dir> --gpu 0
```
