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

import cv2
import numpy as np
from omegaconf import DictConfig

from reality_sr.utils.events import LOGGER
from reality_sr.utils.ops import get_window_position
from .super_resolution import SuperResolutionInferencer

__all__ = [
    "LargeImageSuperResolutionInferencer",
]


class LargeImageSuperResolutionInferencer(SuperResolutionInferencer):
    def __init__(self, config_dict: DictConfig) -> None:
        super().__init__(config_dict)
        self.upscale_factor = config_dict.UPSCALE_FACTOR
        self.window_width = config_dict.WINDOW_WIDTH
        self.window_height = config_dict.WINDOW_HEIGHT
        self.overlap_width = config_dict.OVERLAP_WIDTH
        self.overlap_height = config_dict.OVERLAP_HEIGHT

    def infer(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        windows = []  # List to store the windows
        stride_width = self.window_width - self.overlap_width
        stride_height = self.window_height - self.overlap_height
        for pos_y in range(0, height, stride_height):
            for pos_x in range(0, width, stride_width):
                window_position = get_window_position(pos_x, pos_y, width, height, self.window_width, self.window_height)
                window = image[window_position[1]:window_position[3], window_position[0]:window_position[2]]
                # Resize the window to 4 times its original size
                lr_tensor = self.pre_process(window)
                sr_tensor = self.model(lr_tensor)
                sr_image = self.post_process(sr_tensor)
                windows.append(sr_image)  # Add the window to the list

        # Create a new image that is 4 times the size of the original image
        splicing_image = np.zeros((height * self.upscale_factor, width * self.upscale_factor, 3), dtype=np.uint8)

        i = 0  # Index to keep track of the current window
        for pos_y in range(0, height, stride_height):
            for pos_x in range(0, width, stride_width):
                window_position = get_window_position(pos_x, pos_y, width, height, self.window_width, self.window_height)
                window = windows[i]  # Get the window from the list
                i += 1  # Increment the index
                # Adjust the position to account for the increased size of the windows
                splicing_image[window_position[1] * self.upscale_factor:window_position[1] * self.upscale_factor + window.shape[0],
                window_position[0] * self.upscale_factor:window_position[0] * self.upscale_factor + window.shape[1]] = window

        return splicing_image

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

            # Save image
            output_path = self.output / Path(file_name).name
            cv2.imwrite(str(output_path), sr_image)
            LOGGER.info(f"SR image save to `{output_path}`")
