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
from torch import Tensor, nn

__all__ = [
    "PixShuffleUpsampleBlock",
]


class PixShuffleUpsampleBlock(nn.Module):
    r"""Pixel shuffle upsample block.
    `Enhanced Deep Residual Networks for Single Image Super-Resolution` https://arxiv.org/abs/1707.02921 paper.

    Attributes:
        pix_shuffle_upsample_block (nn.Sequential): The residual convolutional block.
    """

    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(PixShuffleUpsampleBlock, self).__init__()
        self.pix_shuffle_upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, 3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pix_shuffle_upsample_block(x)
