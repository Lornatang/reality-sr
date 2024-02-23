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
from fvcore.nn import FlopCountAnalysis
from torch import nn

__all__ = [
    "get_model_info",
]


def get_model_info(model: nn.Module, image_size: int = 64, device: torch.device = torch.device("cpu")) -> str:
    r"""Get model Params and GFlops.

    Args:
        model (nn.Module): The model whose information is to be retrieved.
        image_size (int, optional): The size of the image. Defaults to 64.
        device (torch.device, optional): The device to use. Defaults to torch.device("cpu").

    Returns:
        str: The information about the model.
    """
    tensor = torch.rand([1, 3, image_size, image_size], device=device)

    params = sum([param.nelement() for param in model.parameters()])
    flops = FlopCountAnalysis(model.to(device), tensor).total()
    params /= 1e6
    flops /= 1e9
    return f"Params: {params:.2f} M, GFLOPs: {flops:.2f} B"
