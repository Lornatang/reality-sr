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
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch.utils.data
from torch import Tensor

from reality_sr.utils.imgproc import image_to_tensor

__all__ = [
    "PairedImageDataset",
]


class PairedImageDataset(torch.utils.data.Dataset):
    r"""Image dataset loading method after registration

    Args:
        paired_gt_images_dir (str or Path): ground truth images after registration
        paired_lr_images_dir (str or Path): registered low-resolution images
    """

    def __init__(
            self,
            paired_gt_images_dir: str | Path,
            paired_lr_images_dir: str | Path,
    ) -> None:
        super(PairedImageDataset, self).__init__()
        # Get a list of all image filenames
        if isinstance(paired_gt_images_dir, str):
            paired_gt_images_dir = Path(paired_gt_images_dir)
        if isinstance(paired_lr_images_dir, str):
            paired_lr_images_dir = Path(paired_lr_images_dir)
        self.paired_gt_image_file_names = [p for p in paired_gt_images_dir.glob("*")]
        self.paired_lr_image_file_names = [p for p in paired_lr_images_dir.glob("*")]

        if len(self.paired_gt_image_file_names) == 0:
            raise ValueError(f"No images found in {paired_gt_images_dir}")
        if len(self.paired_lr_image_file_names) == 0:
            raise ValueError(f"No images found in {paired_lr_images_dir}")
        if len(self.paired_gt_image_file_names) != len(self.paired_lr_image_file_names):
            raise ValueError(f"The number of images in {paired_gt_images_dir} and {paired_lr_images_dir} is different")

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        # read a batch of images
        gt_image = cv2.imread(str(self.paired_gt_image_file_names[batch_index])).astype(np.float32) / 255.
        lr_image = cv2.imread(str(self.paired_lr_image_file_names[batch_index])).astype(np.float32) / 255.

        # BGR image data to RGB image data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image data channel to a data format supported by PyTorch
        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.paired_gt_image_file_names)
