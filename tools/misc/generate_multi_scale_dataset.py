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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Union

import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"]


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--inputs",
        default="./datasets/df2k",
        type=str,
        help="Path to input image directory. Defaults to ``./datasets/df2k``.",
    )
    parser.add_argument(
        "-o", "--output",
        default="./datasets/df2k_multi_scale",
        type=str,
        help="Path to output multi-scale image directory. Defaults to ``./datasets/df2k_multi_scale``.",
    )
    parser.add_argument(
        "--small-size",
        default=384,
        type=int,
        help="Small size of sliding window. Defaults to 384.")
    parser.add_argument(
        "--scale-list",
        nargs="+",
        default=[0.33, 0.5, 0.75],
        type=float,
        help="List of scales to generate multi-scale images. Defaults to [0.33, 0.5, 0.75].",
    )
    parser.add_argument(
        "-s", "--skip",
        action="store_true",
        help="Skip check and conversion for images without corresponding labels.",
    )
    return parser.parse_args()


def validate_image(image_dir: Union[Path, str], skip: bool = False) -> Tuple[List[str], List[str]]:
    image_dir = Path(image_dir)

    valid_image_path_list, invalid_image_path_list = [], []

    image_path_list = [file for file in image_dir.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES]

    for image_path in image_path_list:
        if not skip:
            image = cv2.imread(str(image_path))
            if image is None:
                invalid_image_path_list.append(str(image_path.absolute()))
                continue

        valid_image_path_list.append(str(image_path.absolute()))

    return valid_image_path_list, invalid_image_path_list


def _generate_multi_scale_dataset(
        image_path: Union[Path, str],
        output_dir: Union[Path, str],
        small_size: int,
        scale_list: List[float],
) -> None:
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            LOGGER.warning(f"Reading image failed: {image_path}. Skipping.")
            return

        height, width = image.shape[:2]

        if height < small_size or width < small_size:
            if width < height:
                ratio = height / width
                width = small_size
                height = int(width * ratio)
            else:
                ratio = width / height
                height = small_size
                width = int(height * ratio)
            new_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
            new_image_path = Path(output_dir, f"{image_path.stem}-scale{len(scale_list) + 1}{image_path.suffix}")
            cv2.imwrite(str(new_image_path), new_image)

        for index, scale in enumerate(scale_list):
            new_image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
            new_image_path = Path(output_dir, f"{image_path.stem}-scale{index}{image_path.suffix}")
            cv2.imwrite(str(new_image_path), new_image)

        LOGGER.debug(f"Processed: {image_path.name}.")

    except Exception as e:
        LOGGER.error(f"Error processing {image_path}: {e}.")


def generate_multi_scale_dataset(
        inputs_dir: Union[Path, str],
        output_dir: Union[Path, str],
        small_size: int = 384,
        scale_list: List[float] = None,
        skip: bool = False,
) -> None:
    inputs_dir = Path(inputs_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if scale_list is None:
        scale_list = [0.33, 0.5, 0.75]

    valid_image_path_list, invalid_image_path_list = validate_image(inputs_dir, skip)
    if not valid_image_path_list:
        LOGGER.error("No valid image files found.")
        return
    if invalid_image_path_list:
        LOGGER.warning(f"Invalid or corrupt images: {invalid_image_path_list}")

    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(
                _generate_multi_scale_dataset,
                valid_image_path,
                output_dir,
                small_size,
                scale_list,
            ) for valid_image_path in valid_image_path_list
        ]
        with tqdm(total=len(tasks), desc="Generate multi scale dataset", unit="file") as pbar:
            for _ in as_completed(tasks):
                pbar.update(1)


def main() -> None:
    opts = get_opts()

    generate_multi_scale_dataset(opts.inputs, opts.output, opts.small_size, opts.scale_list, opts.skip)


if __name__ == "__main__":
    main()
