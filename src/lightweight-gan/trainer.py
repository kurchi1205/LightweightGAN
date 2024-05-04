from math import floor
from utils import is_power_of_two
from pathlib import Path
from model import init_GAN
from data import get_data
from accelerate import Accelerator
from torch.optim import Adam


class Trainer:
    def __init__(
        self,
        name = 'default',
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / args.results_dir
        self.models_dir = base_dir / args.models_dir
        self.fid_dir = base_dir / 'fid' / name

        self.config_path = self.models_dir / name / '.config.json'

        assert is_power_of_two(args.image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, args.attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        assert not (args.dual_contrast_loss and args.disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'

        self.image_size = args.image_size
        self.num_image_tiles = args.num_image_tiles

        self.latent_dim = args.latent_dim
        self.fmap_max = args.fmap_max
        self.transparent = args.transparent
        self.greyscale = args.greyscale

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = args.aug_prob
        self.aug_types = args.aug_types

        self.lr = args.lr
        self.optimizer = args.optimizer
        self.num_workers = args.num_workers
        self.ttur_mult = args.ttur_mult
        self.batch_size = args.batch_size
        self.gradient_accumulate_every = args.gradient_accumulate_every

        self.gp_weight = args.gp_weight

        self.evaluate_every = args.evaluate_every
        self.save_every = args.save_every
        self.steps = 0

        self.attn_res_layers = args.attn_res_layers
        self.freq_chan_attn = args.freq_chan_attn

        self.disc_output_size = args.disc_output_size
        self.antialias = args.antialias

        self.dual_contrast_loss = args.dual_contrast_loss

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = args.calculate_fid_every
        self.calculate_fid_num_images = args.calculate_fid_num_images
        self.clear_fid_cache = args.clear_fid_cache

        self.run = None
        self.hparams = args.hparams

        train_dataset, val_dataset, test_dataset = get_data(self.image_size, self.aug_prob)
        
        self.G, self.D, self.GE = init_GAN(
            GAN_params = self.GAN_params,
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            freq_chan_attn = self.freq_chan_attn,
            image_size = self.image_size,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
        )

        if self.optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = self.lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = self.lr * self.ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"


        self.acc_Generator = Accelerator(device_placement=True)
        self.acc_Discriminator = Accelerator(device_placement=True)

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
    