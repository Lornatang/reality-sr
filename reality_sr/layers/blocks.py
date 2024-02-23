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

__all__ = [
    "EnhancedSpatialAttention", "ResidualConvBlock", "ResidualDenseBlock", "ResidualResidualDenseBlock", "ResidualFeatureDistillationBlock",
]


class EnhancedSpatialAttention(nn.Module):
    r"""Residual feature distillation block.
    `Residual Feature Aggregation Network for Image Super-Resolution` https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Residual_Feature_Aggregation_Network_foremaining_Image_Super-Resolution_CVPR_2020_paper.pdf paper.
    """

    def __init__(self, channels: int) -> None:
        super(EnhancedSpatialAttention, self).__init__()
        hidden_channels = channels // 4
        self.conv_1 = nn.Conv2d(channels, hidden_channels, 1, stride=1, padding=0)
        self.cross_conv = nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0)
        self.conv_max = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(hidden_channels, channels, 1, stride=1, padding=0)

        self.max_pool = nn.MaxPool2d(7, 3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)
        cross_conv = self.cross_conv(conv_1)
        conv_2 = self.conv_2(conv_1)

        v_max = self.max_pool(conv_2)
        v_range = self.conv_max(v_max)
        v_range = self.relu(v_range)

        conv_3 = self.conv_3(v_range)
        conv_3 = self.relu(conv_3)
        conv_3_1 = self.conv_3_1(conv_3)
        conv_3_1 = F_torch.interpolate(conv_3_1, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        conv_4 = self.conv_4(conv_3_1 + cross_conv)
        conv_4 = self.sigmoid(conv_4)

        return x * conv_4


class ResidualConvBlock(nn.Module):
    r"""Residual convolutional block.
    `Enhanced Deep Residual Networks for Single Image Super-Resolution` https://arxiv.org/abs/1707.02921 paper.
    
    Attributes:
        rcb (nn.Sequential): The residual convolutional block.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.mul(out, 0.1)
        return torch.add(out, identity)


class ResidualDenseBlock(nn.Module):
    r"""Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv_1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out_1 = self.leaky_relu(self.conv_1(x))
        out_2 = self.leaky_relu(self.conv_2(torch.cat([x, out_1], 1)))
        out_3 = self.leaky_relu(self.conv_3(torch.cat([x, out_1, out_2], 1)))
        out_4 = self.leaky_relu(self.conv_4(torch.cat([x, out_1, out_2, out_3], 1)))
        out_5 = self.identity(self.conv_5(torch.cat([x, out_1, out_2, out_3, out_4], 1)))
        out = torch.mul(out_5, 0.2)
        return torch.add(out, identity)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ResidualResidualDenseBlock(nn.Module):
    r"""Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb_1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        out = torch.mul(out, 0.2)
        return torch.add(out, identity)


class ResidualFeatureDistillationBlock(nn.Module):
    r"""Residual feature distillation block.
    `Residual Feature Distillation Network for Lightweight Image Super-Resolution` https://arxiv.org/abs/2009.11551v1 paper.
    """

    def __init__(self, channels: int) -> None:
        super(ResidualFeatureDistillationBlock, self).__init__()
        self.distilled_channels = self.distilled_channels = channels // 2
        self.remaining_channels = self.remaining_channels = channels

        self.conv_1_distilled = nn.Conv2d(channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_1_remaining = nn.Conv2d(channels, self.remaining_channels, 3, stride=1, padding=1)
        self.conv_2_distilled = nn.Conv2d(self.remaining_channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_2_remaining = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, stride=1, padding=1)
        self.conv_3_distilled = nn.Conv2d(self.remaining_channels, self.distilled_channels, 1, stride=1, padding=0)
        self.conv_3_remaining = nn.Conv2d(self.remaining_channels, self.remaining_channels, 3, padding=1)
        self.conv_4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(self.distilled_channels * 4, channels, 1, stride=1, padding=0)

        self.esa = EnhancedSpatialAttention(channels)
        self.leaky_relu = nn.LeakyReLU(0.05, True)

    def forward(self, x: Tensor) -> Tensor:
        distilled_conv_1 = self.conv_1_distilled(x)
        distilled_conv_1 = self.leaky_relu(distilled_conv_1)
        remaining_conv_1 = self.conv_1_remaining(x)
        remaining_conv_1 = torch.add(remaining_conv_1, x)
        remaining_conv_1 = self.leaky_relu(remaining_conv_1)

        distilled_conv_2 = self.conv_2_distilled(remaining_conv_1)
        distilled_conv_2 = self.leaky_relu(distilled_conv_2)
        remaining_conv_2 = self.conv_2_remaining(remaining_conv_1)
        remaining_conv_2 = torch.add(remaining_conv_2, remaining_conv_1)
        remaining_conv_2 = self.leaky_relu(remaining_conv_2)

        distilled_conv_3 = self.conv_3_distilled(remaining_conv_2)
        distilled_conv_3 = self.leaky_relu(distilled_conv_3)
        remaining_conv_3 = self.conv_3_remaining(remaining_conv_2)
        remaining_conv_3 = torch.add(remaining_conv_3, remaining_conv_2)
        remaining_conv_3 = self.leaky_relu(remaining_conv_3)

        remaining_conv_4 = self.conv_4(remaining_conv_3)
        remaining_conv_4 = self.leaky_relu(remaining_conv_4)

        out = torch.cat([distilled_conv_1, distilled_conv_2, distilled_conv_3, remaining_conv_4], 1)
        out = self.conv_5(out)

        return self.esa(out)
