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
from pathlib import Path

import torch
from torch import nn, Tensor

from reality_sr.utils.checkpoint import load_checkpoint

__all__ = [
    "SuperResolutionBackend",
]


class SuperResolutionBackend(nn.Module):
    def __init__(self, weights_path: str | Path, device: torch.device = None):
        super().__init__()
        assert isinstance(weights_path, str) and Path(weights_path).suffix == ".pkl", f"{Path(weights_path).suffix} format is not supported."
        model = load_checkpoint(weights_path, map_location=device)
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
