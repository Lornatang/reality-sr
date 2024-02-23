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
from copy import deepcopy
from typing import Any

import math
import torch
from torch import nn

from reality_sr.utils.checkpoint import is_parallel

__all__ = [
    "ModelEMA",
]


class ModelEMA(nn.Module):
    """Model Exponential Moving Average

    Reference: `https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage`
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0) -> None:
        r"""Initialize the ModelEMA.

        Args:
            model (nn.Module): The model for which to maintain an EMA.
            decay (float, optional): The decay rate for the moving average. Default is 0.9999.
            updates (int, optional): The number of updates applied to the moving average. Default is 0.
        """
        super().__init__()
        # Create a copy of the model for the EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()

        # Initialize the number of updates and the decay function
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

        # Disable gradients for the parameters of the EMA model
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        r"""Update the EMA for the model.

        Args:
            model (nn.Module): The model for which to update the EMA.
        """
        with torch.no_grad():
            # Increment the number of updates
            self.updates += 1

            # Compute the decay rate
            decay = self.decay(self.updates)

            # Update the EMA weights
            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, item in self.ema.state_dict().items():
                if item.dtype.is_floating_point:
                    item *= decay
                    item += (1 - decay) * state_dict[k].detach()

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ("process_group", "reducer")):
        r"""Update the attributes of the EMA model to match those of the given model.

        Args:
            model (nn.Module): The model whose attributes should be copied.
            include (tuple, optional): A tuple of attribute names to be copied. If empty, all attributes are copied. Default is ().
            exclude (tuple, optional): A tuple of attribute names to be excluded from copying. Default is ("process_group", "reducer").
        """
        self.copy_attr(self.ema, model, include, exclude)

    @staticmethod
    def copy_attr(a: Any, b: Any, include: tuple = (), exclude: tuple = ()) -> None:
        r"""Copy attributes from one instance and set them to another instance.

        Args:
            a (object): The object to which attributes should be copied.
            b (object): The object from which attributes should be copied.
            include (tuple, optional): A tuple of attribute names to be copied. If empty, all attributes are copied. Default is ().
            exclude (tuple, optional): A tuple of attribute names to be excluded from copying. Default is ().
        """
        for k, item in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith("_") or k in exclude:
                continue
            else:
                setattr(a, k, item)
