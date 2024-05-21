from math import floor
from utils import is_power_of_two, image_to_pil, init_folders
from pathlib import Path
from model import init_GAN
from data import get_data
from loss import dual_contrastive_loss, hinge_loss, gen_hinge_loss
from metrics import calculate_fid_given_images
from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import numpy as np
import datetime
import wandb
import logging
import time
from tqdm import tqdm

# wandb.login()
logger = logging.getLogger(__name__)
class Trainer:
    def __init__(
        self,
        args = None,
    ):
        
        self.GAN = None
        self.name = args.name
        self.data_name = args.data
        base_dir = '.'
        self.base_dir = base_dir
        self.models_dir = f"{base_dir}/{args.models_dir}"
        self.fid_dir = f"{base_dir}/'fid'/{self.name}"

        self.config_path = f"{self.models_dir}/{self.name}/.config.json"

        assert is_power_of_two(args.image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        assert all(map(is_power_of_two, args.attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        self.dual_contrast_loss = args.dual_contrast_loss
        assert not (args.dual_contrast_loss and args.disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'
        logger.info("Setting image specifications")
        self.image_size = args.image_size
        self.latent_dim = args.latent_dim
        self.fmap_max = args.fmap_max
        self.transparent = args.transparent
        self.greyscale = args.greyscale

        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        self.aug_prob = args.aug_prob

        self.ttur_mult = args.ttur_mult

        logger.info("Setting training specifications")
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.gradient_accumulate_every = args.gradient_accumulate_every
        self.training_iters = args.training_iters
        self.evaluate_every = args.evaluate_every
        self.checkpoint_every = args.checkpoint_every
        self.steps = 0
        self.resume_from_checkpoint = args.resume_from_checkpoint

        self.attn_res_layers = args.attn_res_layers
        self.freq_chan_attn = args.freq_chan_attn # default values has to be found
        self.disc_output_size = args.disc_output_size


        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        init_folders(self.models_dir)

        self.loader = None

        self.calculate_fid_num_images = args.calculate_fid_num_images

        self.run = None
        logger.info("Loading data")
        self.train_dataset, self.val_dataset, self.test_dataset = get_data(self.data_name, self.image_size, self.aug_prob, args.sample)
        self.train_loader = DataLoader(self.train_dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle = True, drop_last = True, pin_memory = True)
        self.val_loader = DataLoader(self.val_dataset, num_workers = self.num_workers, batch_size = self.val_batch_size, shuffle = False, drop_last = True, pin_memory = True)
        self.test_loader = DataLoader(self.test_dataset, num_workers = self.num_workers, batch_size = self.val_batch_size, shuffle = False, drop_last = True, pin_memory = True)
        

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(
            project = 'lightweight-gan',
            name = self.name + '_' + current_datetime,
            config = {
                "image_size": self.image_size,
                "lr": self.lr,
                "optimizer": self.optimizer,
                "batch_size": self.batch_size,
                "total_steps": self.training_iters
            }
        )

        logger.info("Initializing model")
        self.GAN = init_GAN(
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            freq_chan_attn = self.freq_chan_attn,
            image_size = self.image_size,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
        )
        if self.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint {self.resume_from_checkpoint}")
            model = torch.load(self.resume_from_checkpoint)
            self.GAN.load_state_dict(model['GAN'])

        self.G = self.GAN.G
        self.D = self.GAN.D
        self.GE = self.GAN.GE

        if self.optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = self.lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = self.lr * self.ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"


        self.acc = Accelerator(device_placement=True, gradient_accumulation_steps=self.gradient_accumulate_every)
        print(self.acc.device)


    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
    
    def train(self):
        self.G, self.D, self.train_loader, self.val_loader, self.test_loader, self.G_opt, self.D_opt = self.acc.prepare(self.G, self.D, self.train_loader, self.val_loader, self.test_loader, self.G_opt, self.D_opt)
        self.G.train()
        self.D.train()
        total_disc_loss = torch.zeros([], device=self.acc.device)
        total_gen_loss = torch.zeros([], device=self.acc.device)
        if self.dual_contrast_loss:
            D_loss_fn = dual_contrastive_loss
        else:
            D_loss_fn = hinge_loss

        if self.dual_contrast_loss:
            G_loss_fn = dual_contrastive_loss
            G_requires_calc_real = True
        else:
            G_loss_fn = gen_hinge_loss
            G_requires_calc_real = False

        
        latents = torch.randn(self.batch_size, self.latent_dim).to(self.acc.device)
        with tqdm(total=self.training_iters, desc="Training") as pbar:
            for iter in range(self.training_iters):
                disc_loss_list = []
                gen_loss_list = []
                for image_batch in self.train_loader: 
                    self.D_opt.zero_grad()
                    self.G_opt.zero_grad()
                    real_images = image_batch["image"]
                    with self.acc.accumulate():
                        with torch.no_grad():
                            generated_images = self.G(latents)   
                        fake_output, fake_output_32x32, _ = self.D(generated_images)
                        st = time.time()
                        real_output, real_output_32x32, real_aux_loss = self.D(real_images, calc_aux_loss = True)

                        divergence = D_loss_fn(real_output, fake_output)
                        divergence_32x32 = D_loss_fn(real_output_32x32, fake_output_32x32)
                        disc_loss = divergence + divergence_32x32

                        aux_loss = real_aux_loss
                        disc_loss = disc_loss + aux_loss
                        disc_loss_list.append(disc_loss.item())
                        self.acc.backward(disc_loss)
                        total_disc_loss += divergence
                        self.D_opt.step()

                        generated_images = self.G(latents)
                        fake_output, fake_output_32x32, _ = self.D(generated_images)
                        real_output, real_output_32x32, _ = self.D(real_images) if G_requires_calc_real else (None, None, None)
                        loss = G_loss_fn(fake_output, real_output)
                        loss_32x32 = G_loss_fn(fake_output_32x32, real_output_32x32)

                        gen_loss = loss + loss_32x32
                        gen_loss_list.append(gen_loss.item())
                        self.acc.backward(gen_loss)
                        self.G_opt.step()
                        total_gen_loss += loss
                    
                if iter % 100 == 0:
                    if len(gen_loss_list) == 0:
                        gen_loss_list = [0]
                    logger.info(f"[Iter {iter+1}/{self.training_iters}]    Discriminator loss: {np.mean(disc_loss_list)}, Generator loss: {np.mean(gen_loss_list)}")
                    wandb.log({"Generator loss": np.mean(gen_loss_list), "Discriminator loss": np.mean(disc_loss_list)}, step=iter)

                if iter % 10 == 0 and iter > 20000:
                    self.GAN.EMA()

                if iter <= 25000 and iter % 1000 == 0:
                    self.GAN.reset_parameter_averaging()

                if iter % self.evaluate_every == 0:
                    logger.info("Validating")
                    self.validate(self.val_loader, iter)

                if iter % self.checkpoint_every == 0:
                    logger.info("Saving model")
                    save_data = {
                        'GAN': self.GAN.state_dict(),
                    }
                    num = iter / self.checkpoint_every
                    torch.save(save_data, f"{self.models_dir}/model_{num}.pt")

                pbar.update(1)
        self.d_loss = float(total_disc_loss.item())
        self.g_loss = float(total_gen_loss.item())


    def validate(self, loader, step):
        self.G.eval()
        for iter, image_batch in enumerate(loader):
            real_images = image_batch["image"]
            with torch.no_grad():
                latents = torch.randn(self.val_batch_size, self.latent_dim).to(self.acc.device)
                generated_images = self.G(latents)
            # fid_score = calculate_fid_given_images(generated_images, image_batch, self.batch_size, "cuda")
            pil_generated_images = [image_to_pil(image) for image in generated_images]
            pil_true_images = [image_to_pil(image) for image in real_images]
            # wandb.log({"FID score": fid_score}, step=step)
            wandb.log({"Generated images": [wandb.Image(image) for image in pil_generated_images], "True images": [wandb.Image(image) for image in pil_true_images]}, step=step)
            step += 1
            if iter == self.calculate_fid_num_images - 1:
                break

        
                





