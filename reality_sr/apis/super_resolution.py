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

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from reality_sr.engine.backend import SuperResolutionBackend
from reality_sr.utils.envs import select_device
from reality_sr.utils.events import LOGGER
from reality_sr.utils.general import get_all_filenames
from reality_sr.utils.imgproc import image_to_tensor, tensor_to_image
from reality_sr.utils.torch_utils import get_model_info


class SuperResolutionInferencer(ABC):
    def __init__(self, config_dict: DictConfig) -> None:
        # load inference config
        self.device = select_device(config_dict.DEVICE)
        self.weights_path = config_dict.WEIGHTS_PATH
        self.half = config_dict.HALF
        self.inputs = config_dict.INPUTS
        self.output = config_dict.OUTPUT

        # load model
        self.model = SuperResolutionBackend(self.weights_path, self.device)
        model_info = get_model_info(self.model.model, device=self.device)
        LOGGER.info(f"Model summary: {model_info}")

        if self.half and self.device.type != "cpu":
            self.model.half()
        else:
            self.model.float()
            self.half = False

        # disable gradients
        torch.set_grad_enabled(False)

        # get all images
        self.file_names = get_all_filenames(self.inputs)
        if len(self.file_names) == 0:
            raise ValueError(f"No images found in `{self.inputs}`.")

        # Create a result folder
        self.output = Path(self.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)

    def warmup(self):
        tensor = torch.randn([1, 3, 64, 64]).to(self.device)
        _ = self.model(tensor)

    def pre_process(self, image: np.ndarray) -> Tensor:
        # Convert RGB image channel data to image formats supported by PyTorch
        tensor = image_to_tensor(image, False, False).unsqueeze_(0)

        # Data transfer to the specified device
        return tensor.to(device=self.device, non_blocking=True)

    def post_process(self, tensor: Tensor) -> np.ndarray:
        # Convert the tensor to an image format supported by OpenCV
        image = tensor_to_image(tensor, False, False)

        # Convert the image from RGB to BGR format
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def infer(self, image: np.ndarray) -> np.ndarray:
        lr_tensor = self.pre_process(image)
        sr_tensor = self.model(lr_tensor)
        return self.post_process(sr_tensor)

    def inference(self) -> None:
        for file_name in self.file_names:
            file_name = Path(self.inputs) / file_name
            file_name = str(file_name)
            image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            sr_image = self.infer(image)
            output_path = self.output / Path(file_name).name
            cv2.imwrite(str(output_path), sr_image)
            LOGGER.info(f"SR image save to `{output_path}`")
