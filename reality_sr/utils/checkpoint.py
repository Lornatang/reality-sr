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
import shutil
from pathlib import Path

import torch
from torch import nn

from .events import LOGGER

__all__ = [
    "de_parallel", "is_parallel", "load_checkpoint", "load_state_dict", "save_checkpoint", "strip_optimizer",
]


def de_parallel(model: nn.Module) -> nn.Module:
    """
    De-parallelize a model. Return single-GPU model if model"s type is DP or DDP.

    Args:
        model (nn.Module): The model to de-parallelize.

    Returns:
        nn.Module: The de-parallelized model.
    """
    return model.module if is_parallel(model) else model


def is_parallel(model: nn.Module) -> bool:
    """
    Check if a model is of type DataParallel or DistributedDataParallel.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if the model is of type DataParallel or DistributedDataParallel, False otherwise.
    """
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def load_state_dict(weights_path: str, model: nn.Module, map_location: torch.device) -> nn.Module:
    r"""Load weights from checkpoint file, only assign weights those layers" name and shape are match.

    Args:
        weights_path (str): path to weights file.
        model (nn.Module): model to load weights.
        map_location (torch.device): device to load weights.

    Returns:
        nn.Module: model with weights loaded.
    """
    checkpoint = torch.load(weights_path, map_location=map_location)
    state_dict = checkpoint["model"].float().state_dict()

    # filter out unnecessary keys
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}

    model.load_state_dict(state_dict, strict=False)
    del checkpoint, state_dict, model_state_dict
    return model


def load_checkpoint(weights_path: str | Path, map_location: torch.device | str = torch.device("cpu")) -> torch.nn.Module:
    r"""Load model from a checkpoint file.

    Args:
        weights_path (str or Path): Path to the weights file.
        map_location (torch.device, optional): The device to load the weights to. Defaults to torch.device("cpu").

    Returns:
        torch.nn.Module: The model with the weights loaded.
    """
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)

    if not weights_path.exists():
        LOGGER.error(f"No weights file found at `{weights_path}`")

    LOGGER.info(f"Loading checkpoint from `{weights_path}`")
    checkpoint = torch.load(weights_path, map_location=map_location)
    model = checkpoint["ema" if checkpoint.get("ema") else "model"].float()
    model = model.eval()
    return model


def save_checkpoint(
        checkpoint: dict,
        save_dir: str | Path,
        is_best: bool,
        current_model_name: str | Path,
        best_model_name: str | Path,
        last_model_name: str | Path,
) -> None:
    r"""Save checkpoint to the disk.

    Args:
        checkpoint (dict): The checkpoint to be saved.
        save_dir (str or Path): The directory where to save the checkpoint.
        is_best (bool): Whether this checkpoint is the best so far.
        current_model_name (str or Path, optional): The name of the current model.
        best_model_name (str or Path, optional): The name of the best model.
        last_model_name (str or Path, optional): The name of the model.
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    if isinstance(current_model_name, str):
        current_model_name = Path(current_model_name)
    if isinstance(best_model_name, str):
        best_model_name = Path(best_model_name)

    save_dir.mkdir(parents=True, exist_ok=True)

    current_model_path = save_dir / current_model_name
    last_model_path = save_dir / last_model_name

    torch.save(checkpoint, current_model_path)
    torch.save(checkpoint, last_model_path)

    if is_best:
        best_filename = os.path.join(save_dir, best_model_name)
        shutil.copyfile(current_model_path, best_filename)


def strip_optimizer(checkpoint_path: str | Path, epoch: int) -> None:
    r"""Delete optimizer from saved checkpoint file

    Args:
        checkpoint_path (str | Path): The path to the checkpoint directory.
        epoch (int): The current epoch.
    """
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint directory found at `{checkpoint_path}`")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if checkpoint.get("ema"):
        checkpoint["model"] = checkpoint["ema"]  # replace model with ema
    for k in ["optimizer", "scheduler", "ema", "updates"]:  # keys
        checkpoint[k] = None
    checkpoint["epoch"] = epoch
    checkpoint["model"].half()  # to FP16
    for p in checkpoint["model"].parameters():
        p.requires_grad = False
    torch.save(checkpoint, checkpoint_path)
