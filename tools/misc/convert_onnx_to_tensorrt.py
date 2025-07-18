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

import tensorrt as trt

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--onnx-path",
        type=str,
        required=True,
        help="Path to the onnx model file.",
    )
    parser.add_argument(
        "-o", "--tensorrt-path",
        type=str,
        help="Path to the tensorrt model.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="(1, 3, 64, 64)",
        help="Optimal input shape (NCHW). Defaults to ``(1, 3, 64, 64)``."
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="Workspace size in gb for tensorrt. Defaults to 4."
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable fp16 precision for tensorrt."
    )
    opts = parser.parse_args()

    opts.input_shape = ast.literal_eval(opts.input_shape)
    for shape in [opts.input_shape]:
        if not isinstance(shape, tuple) or len(shape) != 4:
            raise ValueError(f"Invalid shape {shape}, must be 4-dimensional tuple (N, C, H, W)!")

    return opts


def convert_onnx_to_tensorrt(
        onnx_path: Union[Path, str],
        tensorrt_path: Union[Path, str] = None,
        input_shape: Tuple[int, int, int, int] = (1, 3, 88, 137),
        workspace: int = 4,
        half: bool = False,
) -> None:
    if tensorrt_path:
        tensorrt_path = Path(tensorrt_path)
        tensorrt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tensorrt_path = Path(onnx_path).with_suffix(".engine")

    LOGGER.info(f"Exporting '{onnx_path}' model to '{tensorrt_path}'...")

    # Builder configuration.
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = int(workspace * (1 << 30))
    if workspace > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half

    # Read ONNX.
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            LOGGER.error("Failed to parse onnx model.")
            for error in range(parser.num_errors):
                LOGGER.error(f"{error}: {parser.get_error(error)}")
            raise RuntimeError("ONNX parsing failed.")

    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic shapes.
    profile = builder.create_optimization_profile()
    profile.set_shape("images", input_shape, input_shape, input_shape)
    profile.set_shape(network.get_input(0).name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(tensorrt_path, "wb") as f:
        f.write(serialized_engine)
    LOGGER.info(f"TensorRT engine saved to '{tensorrt_path}'!")


def main() -> None:
    opts = get_opts()

    convert_onnx_to_tensorrt(opts.onnx_path, opts.tensorrt_path, opts.input_shape, opts.workspace, opts.half)


if __name__ == "__main__":
    main()
