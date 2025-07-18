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
from pathlib import Path
from typing import Union

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--torch-path",
        type=str,
        required=True,
        help="Path to the torch model.",
    )
    parser.add_argument(
        "-o", "--onnx-path",
        type=str,
        help="Path to the onnx model.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 precision for TensorRT."
    )
    return parser.parse_args()


def get_opset_version() -> int:
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1


def convert_torch_to_onnx(torch_path: Union[Path, str], onnx_path: Union[Path, str] = None, half: bool = False) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TensorRT requires a CUDA-enabled GPU.")
    model = torch.load(torch_path, map_location=device, weights_only=False)["model"].float().eval()

    if onnx_path:
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        onnx_path = Path(torch_path).with_suffix(".onnx")

    LOGGER.info(f"Exporting '{torch_path}' model to '{onnx_path}'...")
    input_tensor = torch.randn((1, 3, 64, 64), dtype=torch.float if half else torch.float32, device=device)
    torch.onnx.export(
        model,
        input_tensor,
        onnx_path,
        opset_version=get_opset_version(),
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch_size", 2: "height", 3: "width"},
            "output0": {0: "batch_size", 2: f"height", 3: f"width"}
        }
    )
    LOGGER.info(f"ONNX engine saved to '{onnx_path}'!")


def main() -> None:
    opts = get_opts()

    convert_torch_to_onnx(opts.torch_path, opts.onnx_path, opts.half)


if __name__ == "__main__":
    main()
