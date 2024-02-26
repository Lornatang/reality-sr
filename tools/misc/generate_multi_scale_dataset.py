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
import argparse
from abc import ABC
from pathlib import Path

import cv2
from torchvision.datasets.folder import IMG_EXTENSIONS


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split image to sub images.")
    parser.add_argument(
        "-i", "--inputs",
        type=Path,
        default="./images",
        help="Path to input image directory. Default: ``./images``",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="./multi_scale_images",
        help="Path to output multi-scale image directory. Default: ``./multi_scale_images``",
    )
    parser.add_argument(
        "--small-size",
        type=int,
        default=384,
        help="Small size of sliding window. Default: 384")
    return parser.parse_args()


class GenerateMultiScaleDataset(ABC):
    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__()
        self.inputs = opts.inputs
        self.output = opts.output
        self.small_size = opts.small_size
        self.scale_list = [0.75, 0.5, 1 / 3]

    def run(self):
        all_image_file_paths = [p for p in self.inputs.glob("*")]

        num_images = len(all_image_file_paths)
        if num_images == 0:
            raise ValueError(f"Directory '{self.inputs}' not have supported image files. Supported image formats: {IMG_EXTENSIONS}")

        self.output.mkdir(parents=True, exist_ok=True)

        for image_file_path in all_image_file_paths:
            image = cv2.imread(str(image_file_path), cv2.IMREAD_UNCHANGED)
            height, width = image.shape[:2]
            if height < self.small_size or width < self.small_size:
                # save the smallest image
                if width < height:
                    ratio = height / width
                    width = self.small_size
                    height = int(width * ratio)
                else:
                    ratio = width / height
                    height = self.small_size
                    width = int(height * ratio)
                new_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                new_image_file_path = Path(self.output, f"{image_file_path.stem}-scale{len(self.scale_list) + 1}{image_file_path.suffix}")
                cv2.imwrite(str(new_image_file_path), new_image)

            for index, scale in enumerate(self.scale_list):
                new_image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
                new_image_file_path = Path(self.output, f"{image_file_path.stem}-scale{index}{image_file_path.suffix}")
                cv2.imwrite(str(new_image_file_path), new_image)


if __name__ == "__main__":
    opts = get_opts()

    app = GenerateMultiScaleDataset(opts)
    app.run()
