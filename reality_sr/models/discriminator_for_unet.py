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
from torch.nn import functional as F_torch
from torch.nn.utils import spectral_norm

__all__ = [
    "DiscriminatorForUNet",
    "discriminator_for_unet",
]


class DiscriminatorForUNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            upsample_method: str = "bilinear",
    ) -> None:
        r"""Discriminator for UNet.

        Args:
            in_channels (int, optional): Number of channels in the input image. Default is 3.
            out_channels (int, optional): Number of channels in the output image. Default is 1.
            channels (int, optional): Number of channels in the intermediate layers. Default is 64.
            upsample_method (str, optional): The upsample method. Default is "bilinear".
        """
        super(DiscriminatorForUNet, self).__init__()
        self.upsample_method = upsample_method

        self.conv_1 = nn.Conv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))
        self.down_block_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, int(channels * 2), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), int(channels * 4), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 8), (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 8), int(channels * 4), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 4), int(channels * 2), (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(int(channels * 2), channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)

        # Down-sampling
        down_1 = self.down_block_1(conv_1)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)

        # Up-sampling
        down_3 = F_torch.interpolate(down_3, scale_factor=2, mode="bilinear", align_corners=False)
        up_1 = self.up_block_1(down_3)

        up_1 = torch.add(up_1, down_2)
        up_1 = F_torch.interpolate(up_1, scale_factor=2, mode="bilinear", align_corners=False)
        up_2 = self.up_block_2(up_1)

        up_2 = torch.add(up_2, down_1)
        up_2 = F_torch.interpolate(up_2, scale_factor=2, mode="bilinear", align_corners=False)
        up_3 = self.up_block_3(up_2)

        up_3 = torch.add(up_3, conv_1)

        out = self.conv_2(up_3)
        out = self.conv_3(out)
        out = self.conv_4(out)

        return out


def discriminator_for_unet(**kwargs) -> DiscriminatorForUNet:
    return DiscriminatorForUNet(**kwargs)
