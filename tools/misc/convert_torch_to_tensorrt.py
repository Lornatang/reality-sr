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
# Copyright (c) AlphaBetter. All rights reserved.
import argparse
import ast
import logging
from pathlib import Path
from typing import Tuple, Union

import torch
import torch_tensorrt

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--torch-path",
        type=str,
        required=True,
        help="Path to the torch model file.",
    )
    parser.add_argument(
        "-o", "--tensorrt-path",
        type=str,
        help="Path to the tensorrt model.",
    )
    parser.add_argument(
        "--min-shape",
        type=str,
        default="(1, 3, 16, 16)",
        help="Minimum input shape for dynamic axes (NCHW). Defaults to ``(1, 3, 16, 16)``."
    )
    parser.add_argument(
        "--opt-shape",
        type=str,
        default="(4, 3, 128, 128)",
        help="Optimal input shape for dynamic axes (NCHW). Defaults to ``(8, 3, 128, 128)``."
    )
    parser.add_argument(
        "--max-shape",
        type=str,
        default="(32, 3, 256, 256)",
        help="Maximum input shape for dynamic axes (NCHW). Defaults to ``(32, 3, 256, 256)``."
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable fp17 precision for tensorrt."
    )
    opts = parser.parse_args()

    opts.min_shape = ast.literal_eval(opts.min_shape)
    opts.opt_shape = ast.literal_eval(opts.opt_shape)
    opts.max_shape = ast.literal_eval(opts.max_shape)
    for shape in [opts.min_shape, opts.opt_shape, opts.max_shape]:
        if not isinstance(shape, tuple) or len(shape) != 4:
            raise ValueError(f"Invalid shape {shape}, must be 4-dimensional tuple (N, C, H, W)!")

    return opts


def convert_torch_to_tensorrt(
        torch_path: Union[Path, str],
        tensorrt_path: Union[Path, str] = None,
        min_shape: Tuple[int, int, int, int] = (1, 3, 16, 16),
        opt_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
        max_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        half: bool = False,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TensorRT requires a CUDA-enabled GPU.")

    if tensorrt_path:
        tensorrt_path = Path(tensorrt_path)
        tensorrt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tensorrt_path = Path(torch_path).with_suffix(".engine")

    LOGGER.info(f"Exporting '{torch_path.resolve()}' model to '{tensorrt_path.resolve()}'...")
    model = torch.load(torch_path, map_location=device, weights_only=False)["model"].eval()

    if half:
        model.half()

    inputs = torch_tensorrt.Input(
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        dtype=torch.half if half else torch.float32,
    )
    tensorrt_engine = torch_tensorrt.compile(model, ir="dynamo", inputs=inputs)
    torch_tensorrt.save(tensorrt_engine, str(tensorrt_path), inputs=inputs)
    LOGGER.info(f"TensorRT engine saved to '{tensorrt_path.resolve()}'!")


def main() -> None:
    opts = get_opts()

    convert_torch_to_tensorrt(opts.torch_path, opts.tensorrt_path, opts.min_shape, opts.opt_shape, opts.max_shape, opts.half)


if __name__ == "__main__":
    main()
