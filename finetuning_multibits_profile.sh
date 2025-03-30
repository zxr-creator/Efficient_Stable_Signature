#!/bin/bash

# Define bit sizes to test
bit_sizes=(8 16 24 32 40 48 56 64)
# Output file for timing results
time_file="multibits.txt"
# Clear the time.txt file at the start (optional, remove if you want to append across runs)
> "$time_file"



# Loop through each bit size and run the command
for num_bits in "${bit_sizes[@]}"; do
    echo "Running finetune_ldm_decoder_noneattack_opt.py with num_bits=${num_bits}..." | tee -a "$time_file"
    nsys_output_file="num_bits_${num_bits}"
    # Run the command and capture time output, redirecting it to time.txt
    # 2>&1 redirects stderr (where time -p writes) to stdout, then >> appends to time.txt
    { /opt/nvidia/nsight-systems/2023.1.2/target-linux-x64/nsys profile --delay=5 -t cuda,nvtx  --output "$nsys_output_file" \
    time -p python finetune_ldm_decoder_noneattack_opt.py \
        --num_keys 1 \
        --ldm_config sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.yaml \
        --ldm_ckpt sd/stable-diffusion-v2-1/v2-1_512-ema-pruned.ckpt \
        --num_bits "$num_bits" \
        --train_dir dataset/COCO/train/ \
        --val_dir dataset/COCO/val/ \
        --use_random_msg_decoder True \
        --msg_decoder_path None \
        --output_dir "output/num_bits_${num_bits}"; } 2>&1 | tee -a "$time_file"
    echo "Finished testing with num_bits=${num_bits}" | tee -a "$time_file"
    echo "---------------------------------------" | tee -a "$time_file"
done

echo "Timing results saved to $time_file"
