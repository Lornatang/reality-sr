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
from typing import Any

from torch import nn

__all__ = [
    "get_window_position", "initialize_weights",
]


def get_window_position(pos_x: int, pos_y: int, image_width: int, image_height: int, window_width: int, window_height: int):
    r"""Get the position of the window.

    Args:
        pos_x (int): The x-coordinate of the window.
        pos_y (int): The y-coordinate of the window.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        window_width (int): The width of the window.
        window_height (int): The height of the window.
    """
    is_right_edge = (image_width - pos_x) < window_width
    is_bottom_edge = (image_height - pos_y) < window_height
    is_right_bottom_edge = is_right_edge and is_bottom_edge

    if is_right_bottom_edge:
        return image_width - window_width, image_height - window_height, image_width, image_height
    if is_bottom_edge:
        return pos_x, image_height - window_height, pos_x + window_width, image_height
    if is_right_edge:
        return image_width - window_width, pos_y, image_width, pos_y + window_height

    return pos_x, pos_y, pos_x + window_width, pos_y + window_height


def initialize_weights(modules: Any):
    r"""Initializes the weights of the model.

     Args:
         modules: The model to be initialized.
     """
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
