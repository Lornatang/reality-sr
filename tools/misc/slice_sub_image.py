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
"""
Crop all images in a folder to sub-images of a specified size, with equal width and height
"""
import argparse
import multiprocessing
import os
from abc import ABC
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm

from reality_sr.utils.ops import check_dir, get_all_filenames


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
        default="./images_sub",
        help="Path to output sub image directory. Default: ``./images_sub``",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=384,
        help="Crop image size from raw image.Default: 384")
    parser.add_argument(
        "--step",
        type=int,
        default=192,
        help="Step size of sliding window.Default: 192")
    parser.add_argument(
        "--thresh-size",
        type=int,
        default=384,
        help="Threshold size. If the remaining image is less than the threshold, it will not be cropped. Default: 384",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="How many threads to open at the same time. Default: 10",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="whether to overwrite existing files. Default: False",
    )
    opts = parser.parse_args()

    return opts


class SliceSubImage(ABC):
    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__()
        self.inputs = opts.inputs
        self.output = opts.output
        self.crop_size = opts.crop_size
        self.step = opts.step
        self.thresh_size = opts.thresh_size
        self.num_workers = opts.num_workers

    def worker(self, file_name: Union[str, Path]) -> None:
        r"""Crop all images in a folder to sub-images of a specified size, with equal width and height.

        Args:
            file_name (str or Path): Image file name.
        """
        if isinstance(file_name, str):
            file_name = Path(file_name)
        image_path = Path(self.inputs) / file_name
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        # Prevent cropped images from crossing the boundary
        image_h, image_w = image.shape[0:2]
        image_h_space = np.arange(0, image_h - self.crop_size + 1, self.step)
        if image_h - (image_h_space[-1] + self.crop_size) > self.thresh_size:
            image_h_space = np.append(image_h_space, image_h - self.crop_size)
        image_w_space = np.arange(0, image_w - self.crop_size + 1, self.step)
        if image_w - (image_w_space[-1] + self.crop_size) > self.thresh_size:
            image_w_space = np.append(image_w_space, image_w - self.crop_size)

        # Cut according to the row and column cycle
        for pos_y in image_h_space:
            for pos_x in image_w_space:
                crop_image = image[pos_y: pos_y + self.crop_size, pos_x:pos_x + self.crop_size, ...]
                crop_image = np.ascontiguousarray(crop_image)
                crop_image_path = os.path.join(self.output, f"{file_name.stem}-{pos_x:04d}_{pos_y:04d}{file_name.suffix}")
                cv2.imwrite(crop_image_path, crop_image)

    def run(self) -> None:
        check_dir(self.inputs)
        image_names = get_all_filenames(self.inputs)
        num_images = len(image_names)
        if num_images == 0:
            raise ValueError(f"Directory '{self.inputs}' not have supported image files. Supported image formats: {IMG_EXTENSIONS}")

        os.makedirs(self.output, exist_ok=opts.exist_ok)

        # Multi-thread cropping image
        pbar = tqdm(total=num_images, unit="image", desc="Split sub image")
        workers_pool = multiprocessing.Pool(self.num_workers)
        for image_name in image_names:
            workers_pool.apply_async(self.worker, args=(image_name,), callback=lambda arg: pbar.update(1))
        workers_pool.close()
        workers_pool.join()
        pbar.close()


if __name__ == "__main__":
    opts = get_opts()

    app = SliceSubImage(opts)
    app.run()
