wandb offline
accelerate launch --config_file accelerate_configs/accelerate_cpu.yaml run_train.py \
--data "julianmoraes/doodles-captions-BLIP" \
--image_size 512 \
--aug_prob 0.6 \
--sample 12 \
--disc_output_size 1 \
--batch_size 2 \
--training_iters 20000 \
--calculate_fid_num_images 2 \
--val_batch_size 1 \
--evaluate_every 100 \
--checkpoint_every 1