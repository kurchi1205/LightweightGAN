wandb online
accelerate launch --config_file accelerate_configs/accelerate_single_gpu.yaml run_train.py \
--data "julianmoraes/doodles-captions-BLIP" \
--aug_prob 0.6 \
--sample 120 \
--dual_contrast_loss \
--disc_output_size 1 \
--batch_size 64 \
--training_iters 20000 \
--calculate_fid_num_images 2 \
--val_batch_size 1 \
--evaluate_every 100