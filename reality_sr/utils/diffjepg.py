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
import itertools

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F_torch

__all__ = [
    "DiffJPEG",
]

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = nn.Parameter(torch.from_numpy(y_table))
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = nn.Parameter(torch.from_numpy(c_table))


class DiffJPEG(nn.Module):
    def __init__(self, differentiable: bool = False) -> None:
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = _jpeg_diff_round
        else:
            rounding = torch.round

        self.compress = _CompressJPEG(rounding)
        self.decompress = _DeCompressJPEG()

    def forward(self, x: Tensor, quality: int or float or Tensor) -> Tensor:
        factor = quality
        if isinstance(factor, (int, float)):
            factor = _calculate_quality_factor(factor)
        else:
            for i in range(factor.size(0)):
                factor[i] = _calculate_quality_factor(factor[i])
        height, width = x.size()[-2:]
        height_pad, width_pad = 0, 0

        if height % 16 != 0:
            height_pad = 16 - height % 16
        if width % 16 != 0:
            width_pad = 16 - width % 16

        x = F_torch.pad(x, (0, width_pad, 0, height_pad), mode="constant", value=0)

        y, cb, cr = self.compress(x, factor)
        out = self.decompress(y, cb, cr, (height + height_pad), (width + width_pad), factor)
        return out[:, :, 0:height, 0:width]


def _calculate_quality_factor(quality: int) -> float:
    r"""Calculate factor corresponding to quality

    Args:
        quality (float): Quality for jpeg compression

    Returns:
        quality_factor (float): Compression factor value
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


def _jpeg_diff_round(x: Tensor) -> Tensor:
    r"""JPEG differentiable

    Args:
        x (Tensor): None.

    Returns:
        jpeg_differentiable (Tensor): None
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


class _RGBToYCbCr(nn.Module):
    r"""Convert RGB to YCbCr"""

    def __init__(self) -> None:
        super(_RGBToYCbCr, self).__init__()
        self.shift = torch.tensor([0., 128., 128.]).to(torch.float32)
        self.matrix = torch.tensor([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]], dtype=torch.float32).T

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        self.matrix = self.matrix.to(device)
        self.shift = self.shift.to(device)

        x = x.permute(0, 2, 3, 1)
        out = torch.tensordot(x, self.matrix, dims=1) + self.shift
        return out.view(x.shape)


class _ChromaSubsampling(nn.Module):
    r"""Chroma subsampling"""

    def __init__(self) -> None:
        super(_ChromaSubsampling, self).__init__()

    @staticmethod
    def forward(x: Tensor) -> [Tensor, Tensor, Tensor]:
        image = x.permute(0, 3, 1, 2).clone()
        cb = F_torch.avg_pool2d(image[:, 1, :, :].unsqueeze(1), (2, 2), (2, 2), count_include_pad=False)
        cr = F_torch.avg_pool2d(image[:, 2, :, :].unsqueeze(1), (2, 2), (2, 2), count_include_pad=False)
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return x[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class _BlockSplitting(nn.Module):
    def __init__(self) -> None:
        super(_BlockSplitting, self).__init__()
        self.k = 8

    def forward(self, x: Tensor) -> Tensor:
        height, _ = x.shape[1:3]
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, height // self.k, self.k, -1, self.k)
        x_transposed = x_reshaped.permute(0, 1, 3, 2, 4)
        return x_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class _DCT8x8(nn.Module):
    def __init__(self) -> None:
        super(_DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, x: Tensor) -> Tensor:
        x = x - 128
        out = self.scale * torch.tensordot(x, self.tensor, dims=2)
        return out.view(x.shape)


class _YQuantize(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = y_table

    def forward(self, x: Tensor, factor: int or float or Tensor) -> Tensor:
        if isinstance(factor, (int, float)):
            out = x.float() / (self.y_table * factor)
        else:
            batch_size = factor.size(0)
            table = self.y_table.expand(batch_size, 1, 8, 8) * factor.view(batch_size, 1, 1, 1)
            out = x.float() / table
        return self.rounding(out)


class _CQuantize(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = c_table

    def forward(self, x: Tensor, factor: int or float or Tensor) -> Tensor:
        if isinstance(factor, (int, float)):
            out = x.float() / (self.c_table * factor)
        else:
            batch_size = factor.size(0)
            table = self.c_table.expand(batch_size, 1, 8, 8) * factor.view(batch_size, 1, 1, 1)
            out = x.float() / table
        out = self.rounding(out)

        return out


class _CompressJPEG(nn.Module):
    def __init__(self, rounding: torch.round or _jpeg_diff_round) -> None:
        super(_CompressJPEG, self).__init__()
        self.l1 = nn.Sequential(_RGBToYCbCr(), _ChromaSubsampling())
        self.l2 = nn.Sequential(_BlockSplitting(), _DCT8x8())
        self.c_quantize = _CQuantize(rounding)
        self.y_quantize = _YQuantize(rounding)

    def forward(self, x: Tensor, factor: Tensor) -> [Tensor, Tensor, Tensor]:
        y, cb, cr = self.l1(x * 255)
        components = {"y": y, "cb": cb, "cr": cr}
        for k in components.keys():
            comp = self.l2(components[k])

            if k in ("cb", "cr"):
                comp = self.c_quantize(comp, factor)
            else:
                comp = self.y_quantize(comp, factor)

            components[k] = comp

        return components["y"], components["cb"], components["cr"]


class _YDeQuantize(nn.Module):
    def __init__(self) -> None:
        super(_YDeQuantize, self).__init__()
        self.y_table = y_table

    def forward(self, x: Tensor, factor: int or float or Tensor) -> Tensor:
        if isinstance(factor, (int, float)):
            out = x * (self.y_table * factor)
        else:
            batch_size = factor.size(0)
            table = self.y_table.expand(batch_size, 1, 8, 8) * factor.view(batch_size, 1, 1, 1)
            out = x * table

        return out


class _CDeQuantize(nn.Module):
    def __init__(self) -> None:
        super(_CDeQuantize, self).__init__()
        self.c_table = c_table

    def forward(self, x: Tensor, factor: int or float or Tensor) -> Tensor:
        if isinstance(factor, (int, float)):
            out = x * (self.c_table * factor)
        else:
            batch_size = factor.size(0)
            table = self.c_table.expand(batch_size, 1, 8, 8) * factor.view(batch_size, 1, 1, 1)
            out = x * table

        return out


class _DeDCT8x8(nn.Module):
    def __init__(self) -> None:
        super(_DeDCT8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos((2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.alpha
        out = 0.25 * torch.tensordot(x, self.tensor, dims=2) + 128
        return out.view(x.shape)


class _DeBlockSplitting(nn.Module):
    def __init__(self) -> None:
        super(_DeBlockSplitting, self).__init__()

    @staticmethod
    def forward(x: Tensor, height: int, width: int) -> Tensor:
        k = 8
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, height // k, width // k, k, k)
        x_transposed = x_reshaped.permute(0, 1, 3, 2, 4)
        return x_transposed.contiguous().view(batch_size, height, width)


class _DeChromaSubsampling(nn.Module):
    def __init__(self) -> None:
        super(_DeChromaSubsampling, self).__init__()

    @staticmethod
    def forward(y: Tensor, cb: Tensor, cr: Tensor) -> Tensor:
        def _repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            return x.view(-1, height * k, width * k)

        cb = _repeat(cb)
        cr = _repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], 3)


class _YCbCrToRGB(nn.Module):
    def __init__(self) -> None:
        super(_YCbCrToRGB, self).__init__()
        self.matrix = torch.tensor([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]], dtype=torch.float32).T
        self.shift = torch.tensor([0, -128., -128.]).to(torch.float32)

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        self.matrix = self.matrix.to(device)
        self.shift = self.shift.to(device)

        out = torch.tensordot(x + self.shift, self.matrix, dims=1)
        return out.view(x.shape).permute(0, 3, 1, 2)


class _DeCompressJPEG(nn.Module):
    def __init__(self) -> None:
        super(_DeCompressJPEG, self).__init__()
        self.c_de_quantize = _CDeQuantize()
        self.y_de_quantize = _YDeQuantize()
        self.de_dct_8x8 = _DeDCT8x8()
        self.de_block_splitting = _DeBlockSplitting()
        self.de_chroma_subsampling = _DeChromaSubsampling()
        self.ycbcr_to_rgb = _YCbCrToRGB()

    def forward(self,
                y: Tensor,
                cb: Tensor,
                cr: Tensor,
                image_height: int,
                image_width: int,
                factor: int) -> Tensor:
        components = {"y": y, "cb": cb, "cr": cr}
        for k in components.keys():
            if k in ("cb", "cr"):
                comp = self.c_de_quantize(components[k], factor)
                height, width = int(image_height / 2), int(image_width / 2)
            else:
                comp = self.y_de_quantize(components[k], factor)
                height, width = image_height, image_width
            comp = self.de_dct_8x8(comp)
            components[k] = self.de_block_splitting(comp, height, width)

        out = self.de_chroma_subsampling(components["y"], components["cb"], components["cr"])
        out = self.ycbcr_to_rgb(out)

        out = torch.min(255 * torch.ones_like(out), torch.max(torch.zeros_like(out), out))
        return out / 255
