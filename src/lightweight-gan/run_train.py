from trainer import Trainer
from argparse import ArgumentParser

def run_train():
    args = ArgumentParser()
    args.add_argument('--name', type = str, default = 'default')
    args.add_argument('--data', type = str, default = 'imagenet2012')
    args.add_argument('--models_dir', type = str, default = 'models')
    args.add_argument('--image_size', type = int, default = 128)
    args.add_argument('--attn_res_layers', type = list, default = [])
    args.add_argument('--latent_dim', type = int, default = 256)
    args.add_argument('--fmap_max', type = int, default = 512)
    args.add_argument('--transparent', action = 'store_true')
    args.add_argument('--greyscale', action = 'store_true')
    args.add_argument('--aug_prob', type = float, default = 0)
    args.add_argument('--lr', type = float, default = 2e-4)
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--num_workers', type = int, default = 4)
    args.add_argument('--batch_size', type = int, default = 16)
    args.add_argument('--val_batch_size', type = int, default = 1)
    args.add_argument('--training_iters', type = int, default = 100)
    args.add_argument('--gradient_accumulate_every', type = int, default = 1)
    args.add_argument('--evaluate_every', type = int, default = 10)
    args.add_argument('--checkpoint_every', type = int, default = 10)
    args.add_argument('--ttur_mult', type = 'int', default = 1)
    args.add_argument('--freq_chan_attn', action = 'store_true')
    args.add_argument('--disc_output_size', type = int, default = 5)
    args.add_argument('--dual_contrast_loss', action = 'store_true')
    args.add_argument('--calculate_fid_num_images', type = int, default = 50000)

    trainer = Trainer(args)
    trainer.train()


