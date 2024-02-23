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
import numpy as np
import torch
from torch import Tensor

__all__ = [
    "rgb_to_ycbcr", "bgr_to_ycbcr", "ycbcr_to_rgb", "ycbcr_to_bgr", "rgb_to_ycbcr_torch", "bgr_to_ycbcr_torch",
]


def rgb_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    r"""Implementation of `rgb2ycbcr()` in Matlab under Python language
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in RGB format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data
    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def bgr_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    r"""Implementation of `bgr2ycbcr()` in Matlab under Python language
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data
    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    r"""Implementation of `ycbcr2rgb()` in Matlab under Python language.
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in YCbCr format

    Returns:
        image (np.ndarray): RGB image array data
    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image,
                      [
                          [0.00456621, 0.00456621, 0.00456621],
                          [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]
                      ]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def ycbcr_to_bgr(image: np.ndarray) -> np.ndarray:
    r"""Implementation of `ycbcr2bgr()` in Matlab under Python language.
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data
    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image,
                      [
                          [0.00456621, 0.00456621, 0.00456621],
                          [0.00791071, -0.00153632, 0],
                          [0, -0.00318811, 0.00625893]
                      ]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    r"""Implementation of `rgb2ycbcr()` in Matlab under PyTorch
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format
    """
    if only_use_y_channel:
        weight = Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    r"""Implementation of `bgr2ycbcr()` in Matlab under PyTorch
    Formula reference: `https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format
    """
    if only_use_y_channel:
        weight = Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor
