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
from typing import Any

import torch.utils.data
from omegaconf import DictConfig
from torch import nn

from reality_sr.data.paired_image_dataset import PairedImageDataset
from reality_sr.evaluation.metrics import PSNR, SSIM, NIQE
from reality_sr.utils.checkpoint import load_checkpoint
from reality_sr.utils.events import LOGGER, AverageMeter, ProgressMeter, Summary
from reality_sr.utils.torch_utils import get_model_info


class Evaler:
    def __init__(self, config_dict: DictConfig, device: torch.device) -> None:
        self.config_dict = config_dict
        self.device = device
        self.upscale_factor = self.config_dict.UPSCALE_FACTOR
        self.dataset_config_dict = self.config_dict.DATASET
        self.eval_config_dict = self.config_dict.EVAL

        self.weights_path = self.eval_config_dict.WEIGHTS_PATH
        self.niqe_weights_path = self.eval_config_dict.NIQE_WEIGHTS_PATH
        self.only_test_y_channel = self.eval_config_dict.ONLY_TEST_Y_CHANNEL

        # IQA model
        self.psnr_model = PSNR(crop_border=self.upscale_factor, only_test_y_channel=self.only_test_y_channel, data_range=1.0)
        self.ssim_model = SSIM(crop_border=self.upscale_factor, only_test_y_channel=self.only_test_y_channel, data_range=255.0)
        self.niqe_model = NIQE(crop_border=self.upscale_factor, niqe_weights_path=self.niqe_weights_path)
        self.psnr_model = self.psnr_model.to(self.device)
        self.ssim_model = self.ssim_model.to(self.device)
        self.niqe_model = self.niqe_model.to(self.device)

    def get_dataloader(self):
        val_datasets = PairedImageDataset(self.dataset_config_dict.VAL_GT_IMAGES_DIR, self.dataset_config_dict.VAL_LR_IMAGES_DIR)
        val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=4,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True)

        return val_dataloader

    def load_model(self) -> nn.Module:
        model = load_checkpoint(self.weights_path, map_location=self.device)
        model_info = get_model_info(model, device=self.device)
        LOGGER.info(f"Model Summary: {model_info}")

        model.eval()
        return model

    def evaluate(self, dataloader: Any = None, model: nn.Module = None, device: torch.device = None) -> tuple[Any, Any, Any]:
        if device is None:
            device = self.device
        if dataloader is None:
            dataloader = self.get_dataloader()
        if model is None:
            model = self.load_model()
            model = model.to(device=device)

        # The information printed by the progress bar
        batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
        psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
        ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
        niqees = AverageMeter("NIQE", ":4.2f", Summary.AVERAGE)
        progress = ProgressMeter(len(dataloader), [batch_time, psnres, ssimes, niqees], prefix=f"Eval: ")

        # Set the model as validation model
        model.eval()

        # Record the start time of verifying a batch
        end = time.time()

        # Disable gradient propagation
        with torch.no_grad():
            for i, (gt, lr) in enumerate(dataloader):
                # Load batches of data
                gt = gt.to(device=device, non_blocking=True)
                lr = lr.to(device=device, non_blocking=True)

                # inference
                sr = model(lr)

                # Calculate the image IQA
                psnr = self.psnr_model(sr, gt)
                ssim = self.ssim_model(sr, gt)
                niqe = self.niqe_model(sr)
                batch_size = lr.size(0)
                psnres.update(psnr.item(), batch_size)
                ssimes.update(ssim.item(), batch_size)
                niqees.update(niqe.item(), batch_size)

                # Record the total time to verify a batch
                batch_time.update(time.time() - end)
                end = time.time()

                # Output a verification log information
                progress.display(i + 1)

        return psnres.avg, ssimes.avg, niqees.avg
