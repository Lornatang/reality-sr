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
import ast
import logging
from pathlib import Path
from typing import Tuple, Union

import tensorrt as trt
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--torch-path",
        type=str,
        required=True,
        help="Path to the PyTorch model file.",
    )
    parser.add_argument(
        "-o", "--trt-path",
        type=str,
        help="Path to the TensorRT model.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="Workspace size in GB for TensorRT. Defaults to 4"
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
        help="Optimal input shape for dynamic axes (NCHW). Defaults to ``(4, 3, 128, 128)``."
    )
    parser.add_argument(
        "--max-shape",
        type=str,
        default="(4, 3, 512, 512)",
        help="Maximum input shape for dynamic axes (NCHW). Defaults to (4, 3, 512, 512)``."
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 precision for TensorRT."
    )
    opts = parser.parse_args()

    opts.min_shape = ast.literal_eval(opts.min_shape)
    opts.opt_shape = ast.literal_eval(opts.opt_shape)
    opts.max_shape = ast.literal_eval(opts.max_shape)
    for shape in [opts.min_shape, opts.opt_shape, opts.max_shape]:
        if not isinstance(shape, tuple) or len(shape) != 4:
            raise ValueError(f"Invalid shape {shape}, must be 4-dimensional tuple (N, C, H, W)!")

    return opts


def get_opset_version() -> int:
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1


def convert_torch_to_trt(
        torch_path: Union[Path, str],
        trt_path: Union[Path, str] = None,
        workspace: int = 4,
        min_shape: Tuple[int, int, int, int] = (1, 3, 16, 16),
        opt_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
        max_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
        half: bool = False,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TensorRT requires a CUDA-enabled GPU.")
    model = torch.load(torch_path, map_location=device, weights_only=False)["model"].eval()

    # Export to ONNX.
    onnx_path = Path(torch_path).with_suffix(".onnx")
    LOGGER.info(f"Exporting ONNX model to {onnx_path}...")
    input_tensor = torch.randn(min_shape, device=device)
    if half:
        model.half()
        input_tensor = input_tensor.half()

    output_tensor = model(input_tensor)
    upsale_factor = int(output_tensor.shape[-1] / input_tensor.shape[-1])
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        opset_version=get_opset_version(),
        do_constant_folding=True,
        input_names=["input0"],
        output_names=["output0"],
        dynamic_axes={
            "input0": {0: "batch_size", 2: "height", 3: "width"},
            "output0": {0: "batch_size", 2: f"height*{upsale_factor}", 3: f"width*{upsale_factor}"}
        }
    )

    # Export to TensorRT.
    if trt_path is None:
        trt_path = Path(torch_path).with_suffix(".engine")
    LOGGER.info(f"Building TensorRT engine to {trt_path}...")

    # Builder configuration.
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = int(workspace * (1 << 30))
    if workspace > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    elif workspace > 0:  # TensorRT 7 & 8
        config.max_workspace_size = workspace
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    half = builder.platform_has_fast_fp16 and half

    # Read ONNX.
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            LOGGER.error("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                LOGGER.error(f"{error}: {parser.get_error(error)}")
            raise RuntimeError("ONNX parsing failed")

    # TensorRT inputs.
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic shapes.
    profile = builder.create_optimization_profile()
    profile.set_shape("input0", min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(trt_path, "wb") as f:
        f.write(serialized_engine)
    LOGGER.info(f"TensorRT engine saved to {trt_path} successfully!")


def main() -> None:
    opts = get_opts()

    convert_torch_to_trt(opts.torch_path, opts.trt_path, opts.workspace, opts.min_shape, opts.opt_shape, opts.max_shape, opts.half)


if __name__ == "__main__":
    main()
