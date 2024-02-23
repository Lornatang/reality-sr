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
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_vision

__all__ = [
    "image_to_tensor", "tensor_to_image", "filter2D_torch", "usm_sharp",
    "USMSharp",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("example_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F_vision.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = (tensor - tensor.min()) / tensor.max() - tensor.min()

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1,3, 256, 256], dtype=torch.float)
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    # Scale the image data from [-1, 1] to [0, 1]
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


def filter2D_torch(image: Tensor, kernel: Tensor) -> Tensor:
    r"""PyTorch implements `cv2.filter2D()`

    Args:
        image (Tensor): Image data, PyTorch data stream format
        kernel (Tensor): Blur kernel data, PyTorch data stream format

    Returns:
        out (Tensor): Image processed with a specific filter on the image
    """
    k = kernel.size(-1)
    batch_size, channels, height, width = image.size()

    if k % 2 == 1:
        image = F_torch.pad(image, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size.")

    ph, pw = image.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        image = image.view(batch_size * channels, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F_torch.conv2d(image, kernel, padding=0).view(batch_size, channels, height, width)
    else:
        image = image.view(1, batch_size * channels, ph, pw)
        kernel = kernel.view(batch_size, 1, k, k).repeat(1, channels, 1, 1).view(batch_size * channels, 1, k, k)

    return F_torch.conv2d(image, kernel, groups=batch_size * channels).view(batch_size, channels, height, width)


def usm_sharp(image: np.ndarray, weight: float = 0.5, radius: int = 50, threshold: int = 10) -> np.ndarray:
    r"""Unsharp mask sharpening

    Args:
        image (np.np.ndarray): Image data, OpenCV data stream format
        weight (float, optional): Sharpening weight. Defaults to 0.5.
        radius (int, optional): Blur radius. Defaults to 50.
        threshold (int, optional): Threshold. Defaults to 10.

    Returns:
        np.np.ndarray: Image processed with a specific filter on the image
    """
    if radius % 2 == 0:
        radius += 1

    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    residual = image - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    out = image + weight * residual
    out = np.clip(out, 0, 1)
    return (soft_mask * out) + ((1 - soft_mask) * image)


class USMSharp(nn.Module):
    def __init__(self, radius: int = 50, sigma: int = 0) -> None:
        r"""Unsharp mask sharpening

        Args:
            radius (int, optional): Blur radius. Defaults to 50.
            sigma (int, optional): Blur sigma. Defaults to 0.
        """
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1

        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor, weight: float = 0.5, threshold: int = 10) -> Tensor:
        usm_blur = filter2D_torch(x, self.kernel)
        residual = x - usm_blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D_torch(mask, self.kernel)
        out = x + weight * residual
        out = torch.clip(out, 0, 1)
        out = soft_mask * out + (1 - soft_mask) * x

        return out
