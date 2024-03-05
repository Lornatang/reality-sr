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
import torch
from torch import Tensor, nn

from reality_sr.utils.color import rgb_to_ycbcr_torch
from reality_sr.utils.ops import check_tensor_shape
from .mse import mse_torch

__all__ = [
    "psnr_torch",
    "PSNR",
]


def psnr_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    r"""PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        psnr_metrics (Tensor): PSNR metrics

    """
    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)

    mse_metrics = mse_torch(raw_tensor, dst_tensor, only_test_y_channel, data_range, eps)
    psnr_metrics = 10 * torch.log10_(data_range ** 2 / mse_metrics)

    return psnr_metrics


class PSNR(nn.Module):
    r"""PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """

        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            psnr_metrics (Tensor): PSNR metrics
        """
        super(PSNR, self).__init__()
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

        psnr_metrics = psnr_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return psnr_metrics
