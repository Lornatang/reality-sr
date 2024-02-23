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
import warnings
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch

from reality_sr.utils.color import rgb_to_ycbcr_torch
from reality_sr.utils.general import check_tensor_shape
from reality_sr.utils.matlab_functions import fspecial_gaussian

__all__ = [
    "ssim_torch",
    "SSIM",
]


def ssim_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        gaussian_kernel_window: Tensor,
        down_sampling: bool = False,
        get_ssim_map: bool = False,
        get_cs_map: bool = False,
        get_weight: bool = False,
        only_test_y_channel: bool = True,
        data_range: float = 255.0,
) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor | Any, Tensor] | Any:
    r"""PyTorch implements SSIM (Structural Similarity) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 255]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 255]
        gaussian_kernel_window (Tensor): Gaussian filter
        down_sampling (bool): Whether to perform down-sampling, default: ``False``
        get_ssim_map (bool): Whether to return SSIM image, default: ``False``
        get_cs_map (bool): whether to return CS image, default: ``False``
        get_weight (bool): whether to return the weight image, default: ``False``
        only_test_y_channel (bool): Whether to test only the Y channel of the image, default: ``True``
        data_range (float, optional): Maximum value range of images. Default: 255.0

    Returns:
        ssim_metrics (Tensor): SSIM metrics
    """

    if data_range != 255.0:
        warnings.warn(f"`data_range` must be 255.0!")
        data_range = 255.0

    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)
        raw_tensor = raw_tensor[:, [0], :, :] * data_range
        dst_tensor = dst_tensor[:, [0], :, :] * data_range
        # Round image data
        raw_tensor = raw_tensor - raw_tensor.detach() + raw_tensor.round()
        dst_tensor = dst_tensor - dst_tensor.detach() + dst_tensor.round()
    else:
        raw_tensor = raw_tensor * data_range
        raw_tensor = raw_tensor - raw_tensor.detach() + raw_tensor.round()
        dst_tensor = dst_tensor * data_range
        dst_tensor = dst_tensor - dst_tensor.detach() + dst_tensor.round()

    gaussian_kernel_window = gaussian_kernel_window.to(raw_tensor.device, dtype=raw_tensor.dtype)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # If the image size is large enough, downsample
    down_sampling_factor = max(1, round(min(raw_tensor.size()[-2:]) / 256))
    if (down_sampling_factor > 1) and down_sampling:
        raw_tensor = F_torch.avg_pool2d(raw_tensor, kernel_size=(down_sampling_factor, down_sampling_factor))
        dst_tensor = F_torch.avg_pool2d(dst_tensor, kernel_size=(down_sampling_factor, down_sampling_factor))

    mu1 = F_torch.conv2d(raw_tensor,
                         gaussian_kernel_window,
                         stride=(1, 1),
                         padding=(0, 0),
                         groups=raw_tensor.shape[1])
    mu2 = F_torch.conv2d(dst_tensor,
                         gaussian_kernel_window,
                         stride=(1, 1),
                         padding=(0, 0),
                         groups=dst_tensor.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F_torch.conv2d(raw_tensor * raw_tensor,
                               gaussian_kernel_window,
                               stride=(1, 1),
                               padding=(0, 0),
                               groups=(dst_tensor * dst_tensor).shape[1]) - mu1_sq
    sigma2_sq = F_torch.conv2d(dst_tensor * dst_tensor,
                               gaussian_kernel_window,
                               stride=(1, 1),
                               padding=(0, 0),
                               groups=(dst_tensor * dst_tensor).shape[1]) - mu2_sq
    sigma12 = F_torch.conv2d(raw_tensor * dst_tensor,
                             gaussian_kernel_window,
                             stride=(1, 1),
                             padding=(0, 0),
                             groups=(dst_tensor * dst_tensor).shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    # Force ssim output to be non-negative to avoid negative results
    cs_map = F_torch.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    ssim_metrics = ssim_map.mean([1, 2, 3])

    if get_ssim_map:
        return ssim_map

    if get_cs_map:
        return ssim_metrics, cs_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / c2) * (1 + sigma2_sq / c2))
        return ssim_map, weights

    return ssim_metrics


class SSIM(nn.Module):
    r"""PyTorch implements SSIM (Structural Similarity) function"""

    def __init__(
            self,
            window_size: int = 11,
            gaussian_sigma: float = 1.5,
            channels: int = 3,
            down_sampling: bool = False,
            get_ssim_map: bool = False,
            get_cs_map: bool = False,
            get_weight: bool = False,
            crop_border: int = 0,
            only_test_y_channel: bool = True,
            **kwargs,
    ) -> None:
        """

        Args:
            window_size (int): Gaussian filter size, must be an odd number, default: ``11``
            gaussian_sigma (float): sigma parameter in Gaussian filter, default: ``1.5``
            channels (int): number of image channels, default: ``3``
            down_sampling (bool): Whether to perform down-sampling, default: ``False``
            get_ssim_map (bool): Whether to return SSIM image, default: ``False``
            get_cs_map (bool): whether to return CS image, default: ``False``
            get_weight (bool): whether to return the weight image, default: ``False``
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            ssim_metrics (Tensor): SSIM metrics

        """
        super(SSIM, self).__init__()
        if only_test_y_channel and channels != 1:
            channels = 1
        self.gaussian_kernel_window = fspecial_gaussian(window_size, gaussian_sigma, channels)
        self.down_sampling = down_sampling
        self.get_ssim_map = get_ssim_map
        self.get_cs_map = get_cs_map
        self.get_weight = get_weight
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        ssim_metrics = ssim_torch(raw_tensor,
                                  dst_tensor,
                                  self.gaussian_kernel_window,
                                  self.down_sampling,
                                  self.get_ssim_map,
                                  self.get_cs_map,
                                  self.get_weight,
                                  self.only_test_y_channel,
                                  **self.kwargs)

        return ssim_metrics
