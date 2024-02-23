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

import cv2
import numpy as np
import torch.utils.data
from torch import Tensor

from reality_sr.utils.imgproc import image_to_tensor
from reality_sr.utils.matlab_functions import image_resize

__all__ = [
    "BicubicImageDataset",
]


class BicubicImageDataset(torch.utils.data.Dataset):
    r"""Image dataset loading method after matlab bicubic

    Args:
        gt_images_dir (str or Path): ground truth images after registration
        upscale_factor (int, optional): Image up scale factor. Defaults to 4
    """

    def __init__(
            self,
            gt_images_dir: str | Path,
            upscale_factor: int = 4,
    ) -> None:
        super(BicubicImageDataset, self).__init__()
        self.upscale_factor = upscale_factor
        # Get a list of all image filenames
        if isinstance(gt_images_dir, str):
            gt_images_dir = Path(gt_images_dir)
        self.gt_image_file_names = [p for p in gt_images_dir.glob("*")]

        if len(self.gt_image_file_names) == 0:
            raise ValueError(f"No images found in {gt_images_dir}")

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        # read a batch of images
        gt_image = cv2.imread(str(self.gt_image_file_names[batch_index])).astype(np.float32) / 255.
        lr_image = image_resize(gt_image, 1 / self.upscale_factor)

        # BGR image data to RGB image data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image data channel to a data format supported by PyTorch
        gt_tensor = image_to_tensor(gt_image, False, False)
        lr_tensor = image_to_tensor(lr_image, False, False)

        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.gt_image_file_names)
