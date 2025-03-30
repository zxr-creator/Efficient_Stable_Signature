conda activate /scr/dataset/yuke/xinrui/conda_env/signature
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
torchrun --nproc_per_node=6  --master_port=2950 main.py \
  --val_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/val --train_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/train/ --output_dir output --eval_freq 5 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 300 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1  \
  --tile True --tile_size 32 --tile_type random_grid 
