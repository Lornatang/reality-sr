# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn, optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from reality_sr.data.degenerated_image_dataset import DegeneratedImageDataset
from reality_sr.data.degradations import degradation_process
from reality_sr.data.paired_image_dataset import PairedImageDataset
from reality_sr.data.transforms import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from reality_sr.layers.ema import ModelEMA
from reality_sr.models import *
from reality_sr.models.losses import FeatureLoss
from reality_sr.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from reality_sr.utils.diffjepg import DiffJPEG
from reality_sr.utils.envs import select_device, set_seed_everything
from reality_sr.utils.events import LOGGER, AverageMeter, ProgressMeter
from reality_sr.utils.general import increment_name
from reality_sr.utils.imgproc import USMSharp
from reality_sr.utils.torch_utils import get_model_info
from .evaler import Evaler


def init_train_env(config_dict: DictConfig) -> [DictConfig, torch.device]:
    r"""Initialize the training environment.

    Args:
        config_dict (DictConfig): The configuration dictionary.

    Returns:
        device (torch.device): The device to be used for training.
    """

    def _resume(config_dict: DictConfig, checkpoint_path: str):
        assert Path(checkpoint_path).is_file(), f"the checkpoint path is not exist: {checkpoint_path}"
        LOGGER.info(f"Resume training from the checkpoint file: `{checkpoint_path}`")
        resume_config_file_path = Path(checkpoint_path).parent.parent / save_config_name
        if resume_config_file_path.exists():
            config_dict = OmegaConf.load(resume_config_file_path)
        else:
            LOGGER.warning(f"Can not find the path of `{Path(checkpoint_path).parent.parent / save_config_name}`, will save exp log to"
                           f" {Path(checkpoint_path).parent.parent}")
            LOGGER.warning(f"In this case, make sure to provide configuration, such as datasets, batch size.")
            config_dict.TRAIN.SAVE_DIR = str(Path(checkpoint_path).parent.parent)
        return config_dict

    # Define the name of the configuration file
    save_config_name = "config.yaml"

    resume_g = config_dict.get("RESUME_G", "")
    resume_d = config_dict.get("RESUME_D", "")

    # Handle the resume training case
    if resume_g:
        checkpoint_path = _resume(config_dict, resume_g)
        config_dict.TRAIN.RESUME_G = checkpoint_path
    elif resume_d:
        checkpoint_path = _resume(config_dict, resume_d)
        config_dict.TRAIN.RESUME_D = checkpoint_path
    else:
        save_dir = config_dict.TRAIN.OUTPUT_DIR / Path(config_dict.EXP_NAME)
        config_dict.TRAIN.SAVE_DIR = str(increment_name(save_dir))
        Path(config_dict.TRAIN.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Select the device for training
    device = select_device(config_dict.DEVICE)

    # Set the random seed
    set_seed_everything(1 + config_dict.TRAIN.RANK, deterministic=(config_dict.TRAIN.RANK == -1))

    # Save the configuration
    OmegaConf.save(config_dict, config_dict.TRAIN.SAVE_DIR / Path(save_config_name))

    return config_dict, device


class Trainer:
    def __init__(self, config_dict: DictConfig, device: torch.device):
        self.config_dict = config_dict
        self.device = device
        self.exp_name = self.config_dict.EXP_NAME
        self.phase = self.config_dict.PHASE
        self.upscale_factor = self.config_dict.UPSCALE_FACTOR
        self.degradation_model_parameters_dict = self.config_dict.get("DEGRADATION_MODEL_PARAMETERS_DICT")
        self.degradation_process_parameters_dict = self.config_dict.get("DEGRADATION_PROCESS_PARAMETERS_DICT")
        self.dataset_config_dict = self.config_dict.DATASET
        self.model_config_dict = self.config_dict.MODEL
        self.train_config_dict = self.config_dict.TRAIN
        self.eval_config_dict = self.config_dict.EVAL

        # ========== Init all config ==========
        # datasets
        self.dataset_train_gt_images_dir = self.dataset_config_dict.TRAIN_GT_IMAGES_DIR
        self.dataset_train_lr_images_dir = self.dataset_config_dict.get("TRAIN_LR_IMAGES_DIR")
        self.dataset_val_gt_images_dir = self.dataset_config_dict.VAL_GT_IMAGES_DIR
        self.dataset_val_lr_images_dir = self.dataset_config_dict.VAL_LR_IMAGES_DIR

        # train
        self.resume_g = self.train_config_dict.get("RESUME_G", "")
        self.resume_d = self.train_config_dict.get("RESUME_D", "")
        # train weights
        self.g_weights_path = self.train_config_dict.get("G_WEIGHTS_PATH", "")
        self.d_weights_path = self.train_config_dict.get("D_WEIGHTS_PATH", "")
        # train dataset
        self.train_image_size = self.train_config_dict.IMAGE_SIZE
        self.train_batch_size = self.train_config_dict.BATCH_SIZE
        self.train_num_workers = self.train_config_dict.NUM_WORKERS

        # train loss
        self.pixel_loss = self.train_config_dict.LOSS.get("PIXEL", "")
        self.feature_loss = self.train_config_dict.LOSS.get("FEATURE", "")
        self.adv_loss = self.train_config_dict.LOSS.get("ADV", "")
        if self.pixel_loss:
            self.pixel_loss_type = self.pixel_loss.get("TYPE", "")
            self.pixel_loss_weight = OmegaConf.to_container(self.pixel_loss.get("WEIGHT", []))
        if self.feature_loss:
            self.feature_loss_type = self.feature_loss.get("TYPE", "")
            self.feature_loss_weight = OmegaConf.to_container(self.feature_loss.get("WEIGHT", []))
        if self.adv_loss:
            self.adv_loss_type = self.adv_loss.get("TYPE", "")
            self.adv_loss_weight = OmegaConf.to_container(self.adv_loss.get("WEIGHT", []))
        # train hyper-parameters
        self.epochs = self.train_config_dict.EPOCHS
        # train setup
        self.local_rank = self.train_config_dict.LOCAL_RANK
        self.rank = self.train_config_dict.RANK
        self.world_size = self.train_config_dict.WORLD_SIZE
        self.dist_url = self.train_config_dict.DIST_URL
        self.save_dir = self.train_config_dict.SAVE_DIR
        # train results
        self.output_dir = self.train_config_dict.OUTPUT_DIR
        self.verbose = self.train_config_dict.VERBOSE

        # ========== Init all objects ==========
        if self.phase not in ["psnr", "gan"]:
            raise NotImplementedError(f"Phase {self.phase} is not implemented. Only support `psnr` and `gan`.")

        # datasets
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        self.num_train_batch = len(self.train_dataloader)
        # Define JPEG compression method and USM sharpening method
        jpeg_operation = DiffJPEG(False)
        usm_sharpener = USMSharp()
        self.jpeg_operation = jpeg_operation.to(device=self.device)
        self.usm_sharpener = usm_sharpener.to(device=self.device)

        # For the PSNR phase
        self.g_model = self.get_g_model()
        self.ema = ModelEMA(self.g_model)
        self.g_optimizer = self.get_g_optimizer()
        self.g_lr_scheduler = self.get_g_lr_scheduler()

        self.start_epoch = 0
        self.current_epoch = 0
        # resume model for training
        if self.resume_g:
            self.g_checkpoint = torch.load(self.resume_g, map_location=self.device)
            if self.g_checkpoint:
                self.resume_g_model()
            else:
                LOGGER.warning(f"Loading state_dict from {self.resume_g} failed, train from scratch...")

        # losses
        self.pixel_criterion = self.define_loss(self.pixel_loss_type)

        # For the GAN phase
        if self.phase == "gan":
            self.d_model = self.get_d_model()
            self.d_optimizer = self.get_d_optimizer()
            self.d_lr_scheduler = self.get_d_lr_scheduler()
            if self.resume_d:
                self.d_checkpoint = torch.load(self.resume_d, map_location=self.device)
                if self.d_checkpoint:
                    self.resume_d_model()
                else:
                    LOGGER.warning(f"Loading state_dict from {self.resume_d} failed, train from scratch...")

            self.feature_criterion = self.define_loss(self.feature_loss_type)
            self.adv_criterion = self.define_loss(self.adv_loss_type)

        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir)

        # training variables
        self.start_time: float = 0.0
        self.batch_time: AverageMeter = AverageMeter("Time", ":6.3f")
        self.data_time: AverageMeter = AverageMeter("Data", ":6.3f")
        self.pixel_losses: AverageMeter = AverageMeter("Pixel loss", ":.4e")
        self.feature_losses: AverageMeter = AverageMeter("Feature loss", ":.4e")
        self.adv_losses: AverageMeter = AverageMeter("Adv loss", ":.4e")
        self.d_gt_probes = AverageMeter("D(GT)", ":6.3f")
        self.d_sr_probes = AverageMeter("D(SR)", ":6.3f")
        self.progress: ProgressMeter = ProgressMeter(
            self.num_train_batch,
            [self.batch_time, self.data_time, self.pixel_losses, self.feature_losses, self.adv_losses, self.d_gt_probes, self.d_sr_probes],
            prefix=f"Epoch: [{self.current_epoch}]")

        # eval for training
        self.evaler = Evaler(config_dict, device)
        # metrics
        self.psnr: float = 0.0
        self.ssim: float = 0.0
        self.niqe: float = 0.0
        self.best_psnr: float = 0.0
        self.best_ssim: float = 0.0
        self.best_niqe: float = 100.0

        # Initialize the mixed precision method
        self.scaler = amp.GradScaler(enabled=self.device.type != "cpu")

        if self.verbose:
            g_model_info = get_model_info(self.g_model, self.train_config_dict.IMAGE_SIZE, self.device)
            LOGGER.info(f"G model: {self.g_model}")
            LOGGER.info(f"G model summary: {g_model_info}")
            if self.phase == "gan":
                d_model_info = get_model_info(self.d_model, self.train_config_dict.IMAGE_SIZE, self.device)
                LOGGER.info(f"D model: {self.d_model}")
                LOGGER.info(f"D model summary: {d_model_info}")

    def get_dataloader(self):
        train_datasets = DegeneratedImageDataset(self.dataset_train_gt_images_dir, self.degradation_model_parameters_dict)
        val_datasets = PairedImageDataset(self.dataset_val_gt_images_dir, self.dataset_val_lr_images_dir)
        # generate dataset iterator
        train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                       batch_size=self.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=self.train_num_workers,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       persistent_workers=True)
        val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True)
        return train_dataloader, val_dataloader

    def get_g_model(self):
        model_g_type = self.model_config_dict.G.TYPE
        if model_g_type == "rrdbnet_x4":
            g_model = rrdbnet_x4(in_channels=self.model_config_dict.G.get("IN_CHANNELS", 3),
                                 out_channels=self.model_config_dict.G.get("OUT_CHANNELS", 3),
                                 channels=self.model_config_dict.G.get("CHANNELS", 64),
                                 growth_channels=self.model_config_dict.G.get("GROWTH_CHANNELS", 32),
                                 num_rrdb=self.model_config_dict.G.get("NUM_RRDB", 23))
        if model_g_type == "edsrnet_x4":
            g_model = edsrnet_x4(in_channels=self.model_config_dict.G.get("IN_CHANNELS", 3),
                                 out_channels=self.model_config_dict.G.get("OUT_CHANNELS", 3),
                                 channels=self.model_config_dict.G.get("CHANNELS", 64),
                                 num_rcb=self.model_config_dict.G.get("NUM_RCB", 16))
        else:
            raise NotImplementedError(f"Model type `{model_g_type}` is not implemented.")
        g_model = g_model.to(self.device)
        if self.g_weights_path:
            LOGGER.info(f"Loading state_dict from {self.g_weights_path} for fine-tuning...")
            g_model = load_state_dict(self.g_weights_path, g_model, map_location=self.device)

        return g_model

    def get_d_model(self):
        model_d_type = self.model_config_dict.D.TYPE
        if model_d_type == "discriminator_for_unet":
            d_model = discriminator_for_unet(in_channels=self.model_config_dict.D.get("IN_CHANNELS", 3),
                                             out_channels=self.model_config_dict.D.get("OUT_CHANNELS", 1),
                                             channels=self.model_config_dict.D.get("CHANNELS", 64),
                                             upsample_method=self.model_config_dict.D.get("UPSAMPLE_METHOD", "bilinear"))

        else:
            raise NotImplementedError(f"Model type `{model_d_type}` is not implemented.")

        return d_model

    def get_g_optimizer(self):
        optim_type = self.train_config_dict.SOLVER.G.OPTIM.TYPE
        if optim_type not in ["adam"]:
            raise NotImplementedError(f"G optimizer {optim_type} is not implemented. Only support `adam`.")

        g_optimizer = optim.Adam(self.g_model.parameters(),
                                 lr=self.train_config_dict.SOLVER.G.OPTIM.LR,
                                 betas=OmegaConf.to_container(self.train_config_dict.SOLVER.G.OPTIM.BETAS))

        LOGGER.info(f"G optimizer: {g_optimizer}")
        return g_optimizer

    def get_d_optimizer(self):
        optim_type = self.train_config_dict.SOLVER.D.OPTIM.TYPE
        if optim_type not in ["adam"]:
            raise NotImplementedError(f"D optimizer {optim_type} is not implemented. Only support `adam`.")

        d_optimizer = optim.Adam(self.d_model.parameters(),
                                 lr=self.train_config_dict.SOLVER.D.OPTIM.LR,
                                 betas=OmegaConf.to_container(self.train_config_dict.SOLVER.D.OPTIM.BETAS))
        LOGGER.info(f"D optimizer: {d_optimizer}")
        return d_optimizer

    def get_g_lr_scheduler(self):
        lr_scheduler_type = self.train_config_dict.SOLVER.G.LR_SCHEDULER.TYPE
        if lr_scheduler_type not in ["step_lr", "multistep_lr", "constant"]:
            raise NotImplementedError(f"G scheduler {lr_scheduler_type} is not implemented. Only support [`step_lr`, `multistep_lr`, `constant`].")

        if lr_scheduler_type == "step_lr":
            g_lr_scheduler = optim.lr_scheduler.StepLR(
                self.g_optimizer,
                step_size=self.train_config_dict.SOLVER.G.LR_SCHEDULER.STEP_SIZE,
                gamma=self.train_config_dict.SOLVER.G.LR_SCHEDULER.GAMMA)
        elif lr_scheduler_type == "multistep_lr":
            g_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.g_optimizer,
                milestones=OmegaConf.to_container(self.train_config_dict.SOLVER.G.LR_SCHEDULER.MILESTONES),
                gamma=self.train_config_dict.SOLVER.G.LR_SCHEDULER.GAMMA)
        else:
            g_lr_scheduler = optim.lr_scheduler.ConstantLR(
                self.g_optimizer,
                factor=self.train_config_dict.SOLVER.G.LR_SCHEDULER.FACTOR,
                total_iters=self.train_config_dict.SOLVER.G.LR_SCHEDULER.TOTAL_ITERS)

        LOGGER.info(f"G lr_scheduler: `{lr_scheduler_type}`")
        return g_lr_scheduler

    def get_d_lr_scheduler(self):
        lr_scheduler_type = self.train_config_dict.SOLVER.D.LR_SCHEDULER.TYPE
        if lr_scheduler_type not in ["step_lr", "multistep_lr", "constant"]:
            raise NotImplementedError(f"G scheduler {lr_scheduler_type} is not implemented. Only support [`step_lr`, `multistep_lr`, `constant`].")

        if lr_scheduler_type == "step_lr":
            d_lr_scheduler = optim.lr_scheduler.StepLR(
                self.d_optimizer,
                step_size=self.train_config_dict.SOLVER.D.LR_SCHEDULER.STEP_SIZE,
                gamma=self.train_config_dict.SOLVER.D.LR_SCHEDULER.GAMMA)
        elif lr_scheduler_type == "multistep_lr":
            d_lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.d_optimizer,
                milestones=OmegaConf.to_container(self.train_config_dict.SOLVER.D.LR_SCHEDULER.MILESTONES),
                gamma=self.train_config_dict.SOLVER.D.LR_SCHEDULER.GAMMA)
        else:
            d_lr_scheduler = optim.lr_scheduler.ConstantLR(
                self.d_optimizer,
                factor=self.train_config_dict.SOLVER.D.LR_SCHEDULER.FACTOR,
                total_iters=self.train_config_dict.SOLVER.D.LR_SCHEDULER.TOTAL_ITERS)

        LOGGER.info(f"D lr_scheduler: `{lr_scheduler_type}`")
        return d_lr_scheduler

    def resume_g_model(self):
        resume_state_dict = self.g_checkpoint["model"].float().state_dict()
        self.g_model.load_state_dict(resume_state_dict, strict=True)
        self.start_epoch = self.g_checkpoint["epoch"] + 1
        self.g_optimizer.load_state_dict(self.g_checkpoint["optimizer"])
        self.g_lr_scheduler.load_state_dict(self.g_checkpoint["scheduler"])
        self.ema.ema.load_state_dict(self.g_checkpoint["ema"].float().state_dict())
        self.ema.updates = self.g_checkpoint["updates"]
        LOGGER.info(f"Resumed g model from epoch {self.start_epoch}")

    def resume_d_model(self):
        resume_state_dict = self.d_checkpoint["model"].float().state_dict()
        self.d_model.load_state_dict(resume_state_dict, strict=True)
        self.start_epoch = self.d_checkpoint["epoch"] + 1
        self.d_optimizer.load_state_dict(self.d_checkpoint["optimizer"])
        self.d_lr_scheduler.load_state_dict(self.d_checkpoint["scheduler"])
        LOGGER.info(f"Resumed d model from epoch {self.start_epoch}")

    def define_loss(self, loss_type: str) -> Any:
        if loss_type not in ["l1_loss", "l2_loss", "feature_loss", "bce_with_logits_loss"]:
            raise NotImplementedError(
                f"Loss type {loss_type} is not implemented. Only support [`l1_loss`, `l2_loss`, `feature_loss`, `bce_with_logits_loss`].")

        if loss_type == "l1_loss":
            criterion = nn.L1Loss()
        elif loss_type == "l2_loss":
            criterion = nn.MSELoss()
        elif loss_type == "feature_loss":
            criterion = FeatureLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        criterion = criterion.to(device=self.device)
        LOGGER.info(f"Loss function: `{criterion}`")
        return criterion

    def degradation_transforms(self, gt: Tensor, gaussian_kernel1: Tensor, gaussian_kernel2: Tensor, sinc_kernel: Tensor) -> [Tensor, Tensor, Tensor]:
        # Get the degraded low-resolution image
        gt_usm, gt, lr = degradation_process(gt,
                                             gaussian_kernel1,
                                             gaussian_kernel2,
                                             sinc_kernel,
                                             self.upscale_factor,
                                             self.degradation_process_parameters_dict,
                                             self.jpeg_operation,
                                             self.usm_sharpener)

        # image data augmentation
        (gt_usm, gt), lr = random_crop_torch([gt_usm, gt], lr, self.train_image_size, self.upscale_factor)
        (gt_usm, gt), lr = random_rotate_torch([gt_usm, gt], lr, self.upscale_factor, [0, 90, 180, 270])
        (gt_usm, gt), lr = random_vertically_flip_torch([gt_usm, gt], lr)
        (gt_usm, gt), lr = random_horizontally_flip_torch([gt_usm, gt], lr)

        return gt_usm, gt, lr

    def train(self):
        try:
            self.before_train_loop()
            for self.current_epoch in range(self.start_epoch, self.epochs):
                self.before_epoch()
                self.train_one_epoch()
                self.after_epoch()

            LOGGER.info(f"Training completed in {(time.time() - self.start_time) / 3600:.3f} hours.")
            g_best_checkpoint_path = Path(self.save_dir) / "weights" / "g_best_checkpoint.pkl"
            g_last_checkpoint_path = Path(self.save_dir) / "weights" / "g_last_checkpoint.pkl"
            d_best_checkpoint_path = Path(self.save_dir) / "weights" / "d_best_checkpoint.pkl"
            d_last_checkpoint_path = Path(self.save_dir) / "weights" / "d_last_checkpoint.pkl"
            strip_optimizer(g_best_checkpoint_path, self.current_epoch)
            strip_optimizer(g_last_checkpoint_path, self.current_epoch)
            if self.phase == "gan":
                strip_optimizer(d_best_checkpoint_path, self.current_epoch)
                strip_optimizer(d_last_checkpoint_path, self.current_epoch)

        except Exception as _:
            LOGGER.error("Training failed.")
            raise
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache()

    def train_psnr(self):
        end = time.time()
        for i, (gt, gaussian_kernel1, gaussian_kernel2, sic_kernel) in enumerate(self.train_dataloader):
            # measure data loading time
            self.data_time.update(time.time() - end)

            gt = gt.to(device=self.device, non_blocking=True)
            gaussian_kernel1 = gaussian_kernel1.to(device=self.device, non_blocking=True)
            gaussian_kernel2 = gaussian_kernel2.to(device=self.device, non_blocking=True)
            sinc_kernel = sic_kernel.to(device=self.device, non_blocking=True)
            loss_pixel_weight = torch.Tensor(self.pixel_loss_weight).to(device=self.device)

            # Initialize the generator gradient
            self.g_model.zero_grad(set_to_none=True)

            # degradation transforms
            gt_usm, gt, lr = self.degradation_transforms(gt, gaussian_kernel1, gaussian_kernel2, sinc_kernel)

            # Mixed precision training
            with amp.autocast(enabled=self.device.type != "cpu"):
                sr = self.g_model(lr)
                loss = self.pixel_criterion(sr, gt_usm)
                loss = torch.sum(torch.mul(loss_pixel_weight, loss))

            # Backpropagation
            self.scaler.scale(loss).backward()
            # update generator weights
            self.scaler.step(self.g_optimizer)
            self.scaler.update()

            # update exponential average model weights
            self.ema.update(self.g_model)

            # Statistical loss value for terminal data output
            batch_size = lr.size(0)
            self.pixel_losses.update(loss.item(), batch_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if i % 100 == 0 or i == self.num_train_batch - 1:
                # Writer Loss to file
                self.tblogger.add_scalar("Train/Pixel_Loss", loss.item(), i + self.current_epoch * self.train_batch_size + 1)
                self.progress.display(i + 1)

    def train_gan(self):
        end = time.time()
        for i, (gt, gaussian_kernel1, gaussian_kernel2, sic_kernel) in enumerate(self.train_dataloader):
            # measure data loading time
            self.data_time.update(time.time() - end)

            gt = gt.to(device=self.device, non_blocking=True)
            gaussian_kernel1 = gaussian_kernel1.to(device=self.device, non_blocking=True)
            gaussian_kernel2 = gaussian_kernel2.to(device=self.device, non_blocking=True)
            sinc_kernel = sic_kernel.to(device=self.device, non_blocking=True)
            pixel_loss_weight = torch.Tensor(self.pixel_loss_weight).to(device=self.device)
            feature_loss_weight = torch.Tensor(self.feature_loss_weight).to(device=self.device)
            adv_loss_weight = torch.Tensor(self.adv_loss_weight).to(device=self.device)

            # Initialize the generator gradient
            self.g_model.zero_grad(set_to_none=True)

            # degradation transforms
            gt_usm, gt, lr = self.degradation_transforms(gt, gaussian_kernel1, gaussian_kernel2, sinc_kernel)

            # Set the real sample label to 1, and the false sample label to 0
            batch_size, _, height, width = gt.shape
            real_label = torch.full([batch_size, 1, height, width], 1.0, dtype=torch.float, device=self.device)
            fake_label = torch.full([batch_size, 1, height, width], 0.0, dtype=torch.float, device=self.device)

            # Start training the generator model
            # During generator training, turn off discriminator backpropagation
            for d_parameters in self.d_model.parameters():
                d_parameters.requires_grad = False

            # Initialize generator model gradients
            self.g_model.zero_grad(set_to_none=True)

            # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
            with amp.autocast(enabled=self.device.type != "cpu"):
                # Use the generator model to generate fake samples
                sr = self.g_model(lr)
                pixel_loss = self.pixel_criterion(sr, gt_usm)
                feature_loss = self.feature_criterion(sr, gt_usm)
                adv_loss = self.adv_criterion(self.d_model(sr), real_label)
                pixel_loss = torch.sum(torch.mul(pixel_loss_weight, pixel_loss))
                feature_loss = torch.sum(torch.mul(feature_loss_weight, feature_loss))
                adv_loss = torch.sum(torch.mul(adv_loss_weight, adv_loss))
                # Calculate the generator total loss value
                g_loss = pixel_loss + feature_loss + adv_loss
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(g_loss).backward()
            # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
            # Finish training the generator model

            # Start training the discriminator model
            # During discriminator model training, enable discriminator model backpropagation
            for d_parameters in self.d_model.parameters():
                d_parameters.requires_grad = True

            # Initialize the discriminator model gradients
            self.d_model.zero_grad(set_to_none=True)

            # Calculate the classification score of the discriminator model for real samples
            with amp.autocast():
                gt_output = self.d_model(gt)
                d_loss_gt = self.adv_criterion(gt_output, real_label)
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(d_loss_gt).backward()

            # Calculate the classification score of the discriminator model for fake samples
            with amp.autocast():
                sr_output = self.d_model(sr.detach().clone())
                d_loss_sr = self.adv_criterion(sr_output, fake_label)
                # Calculate the total discriminator loss value
                d_loss = d_loss_sr + d_loss_gt
            # Call the gradient scaling function in the mixed precision API to
            # bp the gradient information of the fake samples
            self.scaler.scale(d_loss_sr).backward()
            # Improve the discriminator model's ability to classify real and fake samples
            self.scaler.step(self.d_optimizer)
            self.scaler.update()
            # Finish training the discriminator model

            # Calculate the score of the discriminator on real samples and fake samples,
            # the score of real samples is close to 1, and the score of fake samples is close to 0
            d_gt_prob = torch.sigmoid_(torch.mean(gt_output.detach()))
            d_sr_prob = torch.sigmoid_(torch.mean(sr_output.detach()))

            # Statistical accuracy and loss value for terminal data output
            self.pixel_losses.update(pixel_loss.item(), lr.size(0))
            self.feature_losses.update(feature_loss.item(), lr.size(0))
            self.adv_losses.update(adv_loss.item(), lr.size(0))
            self.d_gt_probes.update(d_gt_prob.item(), lr.size(0))
            self.d_sr_probes.update(d_sr_prob.item(), lr.size(0))

            # Calculate the time it takes to fully train a batch of data
            self.batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if i % 100 == 0 or i == self.num_train_batch - 1:
                iters = i + self.current_epoch * self.train_batch_size + 1
                self.tblogger.add_scalar("Train/D_Loss", d_loss.item(), iters)
                self.tblogger.add_scalar("Train/G_Loss", g_loss.item(), iters)
                self.tblogger.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
                self.tblogger.add_scalar("Train/Feature_Loss", feature_loss.item(), iters)
                self.tblogger.add_scalar("Train/Adv_Loss", adv_loss.item(), iters)
                self.tblogger.add_scalar("Train/D(GT)_Prob", d_gt_prob.item(), iters)
                self.tblogger.add_scalar("Train/D(SR)_Prob", d_sr_prob.item(), iters)
                self.progress.display(i + 1)

    def before_train_loop(self):
        LOGGER.info("Training start...")
        self.start_time = time.time()

    def before_epoch(self):
        self.g_model.train()
        # if self.rank != -1:
        #     self.train_dataloader.sampler.set_epoch(self.current_epoch)
        self.g_optimizer.zero_grad()

        # The information printed by the progress bar
        self.batch_time = AverageMeter("Time", ":6.3f")
        self.data_time = AverageMeter("Data", ":6.3f")
        self.pixel_losses = AverageMeter("Pixel loss", ":.4e")
        self.progress = ProgressMeter(self.num_train_batch,
                                      [self.batch_time, self.data_time, self.pixel_losses],
                                      prefix=f"Epoch: [{self.current_epoch}]")

        if self.phase == "gan":
            self.d_model.train()
            self.d_optimizer.zero_grad()
            self.feature_losses = AverageMeter("Feature loss", ":.4e")
            self.adv_losses = AverageMeter("Adv loss", ":.4e")
            self.d_gt_probes = AverageMeter("D(GT)", ":6.3f")
            self.d_sr_probes = AverageMeter("D(SR)", ":6.3f")
            self.progress = ProgressMeter(self.num_train_batch,
                                          [self.batch_time, self.data_time, self.pixel_losses, self.feature_losses, self.adv_losses,
                                           self.d_gt_probes, self.d_sr_probes],
                                          prefix=f"Epoch: [{self.current_epoch}]")

    def train_one_epoch(self):
        if self.phase == "psnr":
            self.train_psnr()
        else:
            self.train_gan()

    def after_epoch(self):
        # update g lr
        self.g_lr_scheduler.step()

        # update attributes for ema model
        self.ema.update_attr(self.g_model)

        self.eval_model()

        # save g ckpt
        is_best = self.psnr > self.best_psnr or self.ssim > self.best_ssim
        ckpt = {
            "model": deepcopy(self.g_model).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.g_optimizer.state_dict(),
            "scheduler": self.g_lr_scheduler.state_dict(),
            "epoch": self.current_epoch,
        }
        save_ckpt_dir = Path(self.save_dir) / "weights"
        save_checkpoint(ckpt, is_best, save_ckpt_dir, model_name="g_last_checkpoint", best_model_name="g_best_checkpoint")

        if self.phase == "gan":
            # update d lr
            self.d_lr_scheduler.step()

            # save d ckpt
            is_best = self.niqe < self.best_niqe
            ckpt = {
                "model": self.d_model,
                "ema": None,
                "updates": None,
                "optimizer": self.d_optimizer.state_dict(),
                "scheduler": self.d_lr_scheduler.state_dict(),
                "epoch": self.current_epoch,
            }
            save_checkpoint(ckpt, is_best, save_ckpt_dir, model_name="d_last_checkpoint", best_model_name="d_best_checkpoint")

        del ckpt

    def eval_model(self):
        self.psnr, self.ssim, self.niqe = self.evaler.evaluate(self.val_dataloader, self.g_model, self.device)
        LOGGER.info(f"Epoch: {self.current_epoch} | PSNR: {self.psnr:.2f} | SSIM: {self.ssim:.4f} | NIQE: {self.niqe:.2f}")
