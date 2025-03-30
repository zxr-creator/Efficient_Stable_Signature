#!/bin/bash

# Define bit sizes to test
/opt/nvidia/nsight-systems/2023.1.2/target-linux-x64/nsys profile --delay=5 -t cuda,nvtx python finetune_ldm_decoder_noneattack_opt.py --num_keys 1 --ldm_config sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.yaml --ldm_ckpt sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.ckpt --msg_decoder_path models/dec_48b_whit.torchscript.pt --train_dir dataset/COCO/train/ --val_dir dataset/COCO/val/ --use_random_msg_decoder False --num_bits 48 --output_dir ./results/num_bits_48


echo "Finished testing with num_bits=48"
echo "---------------------------------------"
