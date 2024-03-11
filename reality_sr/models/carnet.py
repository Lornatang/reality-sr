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

from reality_sr.layers.blocks import CascadingBlock
from reality_sr.layers.upsample import PixShuffleUpsampleBlock
from reality_sr.utils.ops import initialize_weights

__all__ = [
    "CARNet",
    "carnet_x2", "carnet_x3", "carnet_x4",
]


class CARNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            upscale_factor: int = 4,
            image_range: float = 255.,
            mean: Tensor = None,
    ) -> None:
        super(CARNet, self).__init__()
        assert upscale_factor in (2, 3, 4), "Upscale factor should be 2, 3 or 4."

        self.upscale_factor = upscale_factor
        if mean is not None:
            self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)
        self.image_range = image_range

        # Shallow feature extractor
        self.conv_1 = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        
        # Cascading feature extractor
        self.cb_1 = CascadingBlock(channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(int(2 * channels), channels, 1, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.cb_2 = CascadingBlock(channels)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(int(3 * channels), channels, 1, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.cb_3 = CascadingBlock(channels)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(int(4 * channels), channels, 1, stride=1, padding=0),
            nn.ReLU(True)
        )

        # Up-sampling layers
        up_sampling = []
        if (upscale_factor & (upscale_factor - 1)) == 0:  # 2,4,8
            for _ in range(int(math.log(upscale_factor, 2))):
                up_sampling.append(PixShuffleUpsampleBlock(64, 2))
        else:  # 3
            up_sampling.append(PixShuffleUpsampleBlock(64, 3))
        self.up_sampling = nn.Sequential(*up_sampling)
        
        # Final output layer
        self.conv_5 = nn.Conv2d(channels, out_channels, (3, 3), 1, 1)

        self.register_buffer("mean", Tensor([0.4563, 0.4402, 0.4056]).view(1, 3, 1, 1))

        initialize_weights(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        self.mean = self.mean.type_as(x)
        self.mean = self.mean.to(x.device)

        x = x.sub_(self.mean).mul_(self.image_range)

        out = self.conv_1(x)

        cb_1 = self.cb_1(out)
        concat_1 = torch.cat([cb_1, out], 1)
        conv_2 = self.conv_2(concat_1)
        cb_2 = self.cb_2(conv_2)
        concat_2 = torch.cat([concat_1, cb_2], 1)
        conv_3 = self.conv_3(concat_2)
        cb_3 = self.cb_3(conv_3)
        concat_3 = torch.cat([concat_2, cb_3], 1)
        conv_4 = self.conv_4(concat_3)

        out = self.up_sampling(conv_4)
        out = self.conv_5(out)

        out = out.div_(self.image_range).add_(self.mean)
        return torch.clamp_(out, 0.0, 1.0)


def carnet_x2(upscale_factor=2, **kwargs) -> CARNet:
    return CARNet(upscale_factor=upscale_factor, **kwargs)


def carnet_x3(upscale_factor=3, **kwargs) -> CARNet:
    return CARNet(upscale_factor=upscale_factor, **kwargs)


def carnet_x4(upscale_factor=4, **kwargs) -> CARNet:
    return CARNet(upscale_factor=upscale_factor, **kwargs)
