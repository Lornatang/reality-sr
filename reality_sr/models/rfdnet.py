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
from reality_sr.layers.blocks import ResidualFeatureDistillationBlock
from reality_sr.utils.ops import initialize_weights
from torch import Tensor, nn

__all__ = [
    "ResidualFeatureDistillationNet",
    "rfdnet_x2", "rfdnet_x3", "rfdnet_x4", "rfdnet_x8",
]


class ResidualFeatureDistillationNet(nn.Module):
    r"""Residual feature distillation block.
    `Residual Feature Distillation Network for Lightweight Image Super-Resolution` https://arxiv.org/abs/2009.11551v1 paper.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, channels: int = 50, upscale_factor: int = 4) -> None:
        super(ResidualFeatureDistillationNet, self).__init__()
        assert upscale_factor in (2, 3, 4, 8), "Upscale factor should be 2, 3, 4 or 8."
        self.upscale_factor = upscale_factor

        self.conv_1 = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        self.rfdb_1 = ResidualFeatureDistillationBlock(channels)
        self.rfdb_2 = ResidualFeatureDistillationBlock(channels)
        self.rfdb_3 = ResidualFeatureDistillationBlock(channels)
        self.rfdb_4 = ResidualFeatureDistillationBlock(channels)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(int(channels * 4), channels, 1, stride=1, padding=0),
            nn.LeakyReLU(0.05, True),
        )

        self.conv_3 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

        self.up_sampler = nn.Sequential(
            nn.Conv2d(channels, out_channels * (upscale_factor ** 2), 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

        initialize_weights(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)

        rfdb_1 = self.rfdb_1(conv_1)
        rfdb_2 = self.rfdb_2(rfdb_1)
        rfdb_3 = self.rfdb_3(rfdb_2)
        rfdb_4 = self.rfdb_4(rfdb_3)

        x = torch.cat([rfdb_1, rfdb_2, rfdb_3, rfdb_4], 1)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = torch.add(x, conv_1)
        x = self.up_sampler(x)

        return torch.clamp_(x, 0, 1)


def rfdnet_x2(upscale_factor=2, **kwargs) -> ResidualFeatureDistillationNet:
    return ResidualFeatureDistillationNet(upscale_factor=upscale_factor, **kwargs)


def rfdnet_x3(upscale_factor=3, **kwargs) -> ResidualFeatureDistillationNet:
    return ResidualFeatureDistillationNet(upscale_factor=upscale_factor, **kwargs)


def rfdnet_x4(upscale_factor=4, **kwargs) -> ResidualFeatureDistillationNet:
    return ResidualFeatureDistillationNet(upscale_factor=upscale_factor, **kwargs)


def rfdnet_x8(upscale_factor=8, **kwargs) -> ResidualFeatureDistillationNet:
    return ResidualFeatureDistillationNet(upscale_factor=upscale_factor, **kwargs)
