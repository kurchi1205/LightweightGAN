from trainer import Trainer
from argparse import ArgumentParser
from typing import List

def comma_separated_list(value):
    return [int(item) for item in value.split(',')]

def run_train():
    parser = ArgumentParser()
    parser.add_argument('--name', type = str, default = 'default')
    parser.add_argument('--data', type = str, default = 'imagenet2012')
    parser.add_argument('--sample', type = int, default = None)
    parser.add_argument('--models_dir', type = str, default = 'models')
    parser.add_argument('--image_size', type = int, default = 128)
    parser.add_argument('--attn_res_layers', type = comma_separated_list, default = None)
    parser.add_argument('--latent_dim', type = int, default = 256)
    parser.add_argument('--fmap_max', type = int, default = 512)
    parser.add_argument('--transparent', action = 'store_true')
    parser.add_argument('--greyscale', action = 'store_true')
    parser.add_argument('--aug_prob', type = float, default = 0)
    parser.add_argument('--lr', type = float, default = 2e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--val_batch_size', type = int, default = 1)
    parser.add_argument('--training_iters', type = int, default = 100)
    parser.add_argument('--gradient_accumulate_every', type = int, default = 1)
    parser.add_argument('--evaluate_every', type = int, default = 10)
    parser.add_argument('--checkpoint_every', type = int, default = 10)
    parser.add_argument('--resume_from_checkpoint', type = str, default = None)
    parser.add_argument('--ttur_mult', type = int, default = 1)
    parser.add_argument('--freq_chan_attn', action = 'store_true')
    parser.add_argument('--disc_output_size', type = int, default = 5)
    parser.add_argument('--dual_contrast_loss', action = 'store_true')
    parser.add_argument('--calculate_fid_num_images', type = int, default = 50000)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__=='__main__':
    run_train()
