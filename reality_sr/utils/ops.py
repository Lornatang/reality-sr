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
import logging
import os
from pathlib import Path
from typing import Any, Union

from torch import Tensor, nn
from torchvision.datasets.folder import IMG_EXTENSIONS

__all__ = [
    "check_dir", "check_tensor_shape", "get_all_filenames", "get_window_position", "increment_name", "initialize_weights",
]

logger = logging.getLogger(__name__)


def check_dir(dir_path: Union[str, Path]) -> None:
    r"""Check if the input directory exists and is a directory.

    Args:
        dir_path (str or Path): Input directory path.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileExistsError(f"Input directory '{dir_path}' not exists.")

    if not dir_path.is_dir():
        raise TypeError(f"'{dir_path}' is not a directory.")


def check_tensor_shape(raw_tensor: Tensor, dst_tensor: Tensor):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or Tensor): reference image tensor flow, RGB format, data range [0, 1]
    """

    # Check if the tensor scale is consistent
    assert raw_tensor.shape == dst_tensor.shape, f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


def get_all_filenames(path: str | Path, image_extensions: tuple = None) -> list:
    r"""Get all file names in the input folder.

    Args：
        path (str or Path): Input directory path.
        image_extensions (tuple): Supported image format. Default: ``None``.
    """
    if isinstance(path, str):
        path = Path(path)

    if image_extensions is None:
        image_extensions = IMG_EXTENSIONS

    # Only get file names with specified extensions
    file_paths = path.iterdir()
    file_names = [p.name for p in file_paths if p.suffix in image_extensions]

    return file_names


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


def increment_name(path: str | Path) -> Path:
    r"""Increase the save directory"s id if the path already exists.

    Args:
        path (str or Path): The path to the directory.

    Returns:
        Path: The updated path with an incremented id if the original path already exists.
    """
    # Convert the path to a Path object
    if isinstance(path, str):
        path = Path(path)
    separator = ""

    # If the path already exists, increment the id
    if path.exists():
        # If the path is a file, remove the suffix
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        new_path = path
        for number in range(1, 9999):
            new_path = f"{path}{separator}{number}{suffix}"
            if not os.path.exists(new_path):
                break
        path = Path(new_path)

    return path


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
