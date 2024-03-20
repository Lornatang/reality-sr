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
from abc import ABC
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch import Tensor

from reality_sr.engine.backend import SuperResolutionBackend
from reality_sr.utils.envs import select_device
from reality_sr.utils.events import LOGGER
from reality_sr.utils.imgproc import tensor_to_image
from reality_sr.utils.torch_utils import get_model_info

__all__ = [
    "SuperResolutionInference",
]


class SuperResolutionInference(ABC):
    def __init__(self, weights_path: Union[str, Path], device: str) -> None:
        self.device = select_device(device)
        self.model = SuperResolutionBackend(weights_path, self.device)
        model_info = get_model_info(self.model.model, device=self.device)
        LOGGER.info(f"Model summary: {model_info}")

        # disable gradients calculation
        torch.set_grad_enabled(False)

        # warmup
        self.model(torch.zeros(1, 3, 64, 64, device=self.device).type_as(next(self.model.parameters())))

    def __call__(self, inputs: Union[str, list[str]], batch_size: int, save_dir: Union[str, Path]) -> None:
        if isinstance(inputs, str):
            inputs = [inputs]
        num_inputs = len(inputs)
        if batch_size > num_inputs:
            LOGGER.warning(f"Batch size {batch_size} is larger than number of inputs {num_inputs}")
            batch_size = num_inputs
        assert save_dir is not None, "save_dir cannot be None"

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for batch_index in range(0, num_inputs, batch_size):
            batch_input_image_list = inputs[batch_index:batch_index + batch_size]
            tensor = self.pre_process(batch_input_image_list)
            predictions = self.inferences(tensor)
            self.post_process(predictions, batch_input_image_list, save_dir)

    @staticmethod
    def _load_image(image_path: Union[str, Path]) -> Tensor:
        if isinstance(image_path, Path):
            image_path = str(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def pre_process(self, image_path_list: list[str]) -> Tensor:
        tensor = []
        for image_path in image_path_list:
            image = self._load_image(image_path)
            tensor.append(image)
        tensor = torch.stack(tensor, dim=0)
        return tensor.to(device=self.device)

    def inferences(self, tensor: Tensor) -> Tensor:
        return self.model(tensor)

    @staticmethod
    def post_process(predictions: Tensor, image_path_list: list[str], save_dir: Path) -> None:
        for i, (prediction, image_path) in enumerate(zip(predictions, image_path_list)):
            save_image_path = str((Path(save_dir).absolute().resolve() / Path(image_path).name))
            save_image = tensor_to_image(prediction, False, False)
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            LOGGER.info(f"SR image save to '{save_image_path}'")
            cv2.imwrite(save_image_path, save_image)
