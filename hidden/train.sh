conda activate /scr/dataset/yuke/xinrui/conda_env/signature
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --master_port=29564 main.py \
  --val_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/val --train_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/train/ --output_dir output/tile64 --eval_freq 10 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 100 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --tile True --tile_size 64 --tile_type grid 
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --master_port=29565 main.py \
  --val_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/val --train_dir /scr/dataset/yuke/xinrui/stable_signature/dataset/COCO/train/ --output_dir output/tile80 --eval_freq 10 \
  --img_size 256 --num_bits 48  --batch_size 16 --epochs 100 \
  --scheduler CosineLRScheduler,lr_min=1e-6,t_initial=300,warmup_lr_init=1e-6,warmup_t=5  --optimizer Lamb,lr=2e-2 \
  --p_color_jitter 0.0 --p_blur 0.0 --p_rot 0.0 --p_crop 1.0 --p_res 1.0 --p_jpeg 1.0 \
  --scaling_w 0.3 --scale_channels False --attenuation none \
  --loss_w_type bce --loss_margin 1 \
  --tile True --tile_size 80 --tile_type grid 