from math import floor
from utils import is_power_of_two
from pathlib import Path
from model import LightweightGAN


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
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = args.batch_size
        self.gradient_accumulate_every = args.gradient_accumulate_every

        self.gp_weight = gp_weight

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

        # self.is_ddp = is_ddp
        # self.is_main = rank == 0
        # self.rank = rank
        # self.world_size = world_size

        # self.syncbatchnorm = is_ddp

        # self.load_strict = args.load_strict

        # self.amp = amp
        # self.G_scaler = GradScaler(enabled = self.amp)
        # self.D_scaler = GradScaler(enabled = self.amp)

        self.run = None
        self.hparams = args.hparams

        # if self.is_main and use_aim:
        #     try:
        #         import aim
        #         self.aim = aim
        #     except ImportError:
        #         print('unable to import aim experiment tracker - please run `pip install aim` first')

        #     self.run = self.aim.Run(run_hash=aim_run_hash, repo=aim_repo)
        #     self.run['hparams'] = hparams

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
    
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # set some global variables before instantiating GAN

        # global norm_class
        global Blur

        # norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        Blur = nn.Identity if not self.antialias else Blur

        # handle bugs when
        # switching from multi-gpu back to single gpu

        # if self.syncbatchnorm and not self.is_ddp:
        #     import torch.distributed as dist
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] = '12355'
        #     dist.init_process_group('nccl', rank=0, world_size=1)

        # instantiate GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr = self.lr,
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            freq_chan_attn = self.freq_chan_attn,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
            antialias = self.antialias
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)