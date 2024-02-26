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

from reality_sr.layers.blocks import ResidualResidualDenseBlock
from reality_sr.utils.ops import initialize_weights

__all__ = [
    "RRDBNet",
    "rrdbnet_x2", "rrdbnet_x3", "rrdbnet_x4", "rrdbnet_x8",
]


class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        super(RRDBNet, self).__init__()
        assert upscale_factor in (2, 3, 4, 8), "Upscale factor should be 2, 3, 4 or 8."
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer
        self.conv_1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv_2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Up-sampling convolutional layer
        if self.upscale_factor == 2 or self.upscale_factor == 3:
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        elif self.upscale_factor == 4:
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        else:  # 8
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )

        # Output layer
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv_4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        initialize_weights(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)
        x = self.trunk(conv_1)
        x = self.conv_2(x)
        x = torch.add(conv_1, x)

        if self.upscale_factor == 2:
            x = self.up_sampling_1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        elif self.upscale_factor == 3:
            x = self.up_sampling_1(F_torch.interpolate(x, scale_factor=3, mode="nearest"))
        elif self.upscale_factor == 4:
            x = self.up_sampling_1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
        else:  # 8
            x = self.up_sampling_1(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_2(F_torch.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_3(F_torch.interpolate(x, scale_factor=2, mode="nearest"))

        x = self.conv_3(x)
        x = self.conv_4(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x


def rrdbnet_x2(upscale_factor=2, **kwargs) -> RRDBNet:
    return RRDBNet(upscale_factor=upscale_factor, **kwargs)


def rrdbnet_x3(upscale_factor=3, **kwargs) -> RRDBNet:
    return RRDBNet(upscale_factor=upscale_factor, **kwargs)


def rrdbnet_x4(upscale_factor=4, **kwargs) -> RRDBNet:
    return RRDBNet(upscale_factor=upscale_factor, **kwargs)


def rrdbnet_x8(upscale_factor=8, **kwargs) -> RRDBNet:
    return RRDBNet(upscale_factor=upscale_factor, **kwargs)
