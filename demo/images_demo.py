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
from pathlib import Path

from reality_sr.apis.image_inferencer import SuperResolutionImageInferencer
from reality_sr.utils.ops import get_all_filenames


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-path",
        type=Path,
        default="results/pretrained_models/realesrgan_x4-df2k.pkl",
        help="path to model weights file. Defaults to 'results/pretrained_models/realesrgan_x4-df2k.pkl'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Whether to run inference on CPU or GPU. Defaults to '0'",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default="demo/data",
        help="Directory to save output images, defaults to 'demo/data'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="How many images to process at once. Defaults to 2",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default="demo/output",
        help="Directory to save output images. Defaults to 'demo/output'",
    )
    return parser.parse_args()


def main() -> None:
    opts = get_opts()

    inferencer = SuperResolutionImageInferencer(opts.weights_path, opts.device)

    image_name_list = get_all_filenames(opts.inputs_dir)
    image_path_list = [str(Path(opts.inputs_dir).absolute().resolve() / image_name) for image_name in image_name_list]
    inferencer(image_path_list, opts.batch_size, opts.save_dir)


if __name__ == "__main__":
    main()
