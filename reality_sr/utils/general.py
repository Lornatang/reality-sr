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
from typing import Union

from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS

__all__ = [
    "check_dir", "check_tensor_shape", "get_all_filenames", "increment_name",
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
