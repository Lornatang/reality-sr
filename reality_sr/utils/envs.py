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
import os
import random
from typing import Tuple

import numpy as np
import torch

from .events import LOGGER

__all__ = [
    "get_envs", "select_device", "set_seed_everything",
]


def get_envs() -> Tuple[int, int, int]:
    """
    Get PyTorch needed environments from system environments.

    Returns:
        Tuple[int, int, int]: A tuple containing local_rank, rank, and world_size.
    """
    # Get the local rank from the environment variables. If not set, default to -1
    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Get the rank from the environment variables. If not set, default to -1
    rank = int(os.getenv("RANK", -1))

    # Get the world size from the environment variables. If not set, default to 1
    world_size = int(os.getenv("WORLD_SIZE", 1))

    return local_rank, rank, world_size


def select_device(device: str) -> torch.device:
    r"""Set devices' information to the program.

    Args:
        device (str): A string, like "cpu" or "0" or "1,2,3,4"

    Returns:
        torch.device: The selected device.
    """
    # If the device is set to "cpu", set the CUDA_VISIBLE_DEVICES environment variable to "-1" and log that the CPU is being used for training
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        LOGGER.info("Using CPU... ")

    # If a device is specified, set the `CUDA_VISIBLE_DEVICES` environment variable to the specified device and log the number of GPUs being used for
    # training
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        assert torch.cuda.is_available()
        num_device = len(device.strip().split(","))
        LOGGER.info(f"Using {num_device} GPU... ")

    # Check if CUDA is available and the device is not set to "cpu"
    cuda = device != "cpu" and torch.cuda.is_available()

    # Set the device to "cuda:0" if CUDA is available, otherwise set it to "cpu"
    device = torch.device("cuda:0" if cuda else "cpu")

    return device


def set_seed_everything(seed: int, deterministic: bool = False) -> None:
    r"""Set the seed for all possible sources of randomness to ensure reproducibility.

    Args:
        seed (int): The seed value to be set.
        deterministic (bool, optional): If True, the function will set CuDNN backend to deterministic.
            Setting this to True may slow down the training, but it ensures reproducibility. Default is False.
    """
    # Set the seed for Python"s random module
    random.seed(seed)

    # Set the seed for Numpy random module
    np.random.seed(seed)

    # Set the seed for PyTorch"s random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # If deterministic is set to True, make CuDNN deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
