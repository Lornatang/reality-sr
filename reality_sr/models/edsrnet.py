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
import math

import torch
from torch import Tensor, nn

from reality_sr.layers.blocks import ResidualConvBlock
from reality_sr.layers.upsample import PixShuffleUpsampleBlock
from reality_sr.utils.ops import initialize_weights

__all__ = [
    "EDSRNet",
    "edsrnet_x2", "edsrnet_x3", "edsrnet_x4",
]


class EDSRNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 16,
            upscale_factor: int = 4,
            image_range: float = 255.,
            mean: Tensor = None,
    ) -> None:
        super(EDSRNet, self).__init__()
        assert upscale_factor in (2, 3, 4), "Upscale factor should be 2, 3 or 4."

        self.upscale_factor = upscale_factor
        if mean is not None:
            self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        self.image_range = image_range

        # First layer
        self.conv_1 = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        # Residual blocks
        trunk = []
        for _ in range(num_rcb):
            trunk.append(ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv_2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

        # Up-sampling layers
        up_sampling = []
        if (upscale_factor & (upscale_factor - 1)) == 0:  # 2,4,8
            for _ in range(int(math.log(upscale_factor, 2))):
                up_sampling.append(PixShuffleUpsampleBlock(64, 2))
        else:  # 3
            up_sampling.append(PixShuffleUpsampleBlock(64, 3))
        self.up_sampling = nn.Sequential(*up_sampling)

        # Final output layer
        self.conv_3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        initialize_weights(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        self.mean = self.mean.type_as(x)
        self.mean = self.mean.to(x.device)

        x = x.sub_(self.mean).mul_(self.image_range)

        conv_1 = self.conv_1(x)
        x = self.trunk(conv_1)
        x = self.conv_2(x)
        x = torch.add(x, conv_1)
        x = self.up_sampling(x)
        x = self.conv_3(x)
        x = x.div_(self.image_range).add_(self.mean)
        return torch.clamp_(x, 0.0, 1.0)


def edsrnet_x2(upscale_factor=2, **kwargs) -> EDSRNet:
    return EDSRNet(upscale_factor=upscale_factor, **kwargs)


def edsrnet_x3(upscale_factor=3, **kwargs) -> EDSRNet:
    return EDSRNet(upscale_factor=upscale_factor, **kwargs)


def edsrnet_x4(upscale_factor=4, **kwargs) -> EDSRNet:
    return EDSRNet(upscale_factor=upscale_factor, **kwargs)
