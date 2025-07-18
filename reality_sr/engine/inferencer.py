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
import json
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tensorrt as trt
import torch
from reality_sr.utils.events import LOGGER
from torch import nn

__all__ = [
    "TensorRTInferencer",
]


class TensorRTInferencer(nn.Module):
    def __init__(self, tensorrt_path: Union[Path, str], device: torch.device = torch.device("cuda:0")) -> None:
        super().__init__()
        if device.type != "cuda":
            raise ValueError("TensorRTInferencer requires a CUDA device.")

        self.device = device

        logger = trt.Logger(trt.Logger.WARNING)
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        with open(str(tensorrt_path), "rb") as file, trt.Runtime(logger) as runtime:
            try:
                metadata_length = int.from_bytes(file.read(4), byteorder="little")
                metadata = json.loads(file.read(metadata_length).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
            except (UnicodeDecodeError, ValueError):
                file.seek(0)
            self.model = runtime.deserialize_cuda_engine(file.read())

        try:
            self.context = self.model.create_execution_context()
        except Exception as e:
            raise RuntimeError(f"Create execution context failed (possibly version incompatible): {e}") from e

        self.input_names = []
        self.output_names = []
        self.bindings = OrderedDict()

        for i in range(self.model.num_io_tensors):
            name = self.model.get_tensor_name(i)
            dtype = trt.nptype(self.model.get_tensor_dtype(name))
            is_input = self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self.input_names.append(name)
                if -1 in self.model.get_tensor_shape(name):
                    opt_shape = self.model.get_tensor_profile_shape(name, 0)[1]  # (min, opt, max) -> opt
                    self.context.set_input_shape(name, tuple(opt_shape))
            else:
                self.output_names.append(name)
            shape = tuple(self.context.get_tensor_shape(name))
            tensor = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, tensor, int(tensor.data_ptr()))

        self.binding_address = OrderedDict((name, binding.ptr) for name, binding in self.bindings.items())
        self.main_input_name = self.input_names[0] if self.input_names else ""

    def inference(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if x.shape != self.bindings[self.main_input_name].shape:
            self.context.set_input_shape(self.main_input_name, x.shape)
            self.bindings[self.main_input_name] = self.bindings[self.main_input_name]._replace(shape=x.shape)

            for name in self.output_names:
                new_shape = tuple(self.context.get_tensor_shape(name))
                old_binding = self.bindings[name]
                if np.prod(new_shape) > np.prod(old_binding.shape):
                    new_tensor = torch.from_numpy(np.empty(new_shape, dtype=old_binding.dtype)).to(self.device)
                    self.bindings[name] = old_binding._replace(shape=new_shape, data=new_tensor, ptr=int(new_tensor.data_ptr()))
                    self.binding_address[name] = self.bindings[name].ptr
                else:
                    old_binding.data.resize_(new_shape)

        assert x.shape == self.bindings[
            self.main_input_name].shape, f"Input shape mismatch: expected {self.bindings[self.main_input_name].shape}, got {x.shape}"

        self.binding_address[self.main_input_name] = int(x.data_ptr())
        self.context.execute_v2(list(self.binding_address.values()))
        if len(self.output_names) == 1:
            return self.bindings[self.output_names[0]].data
        return [self.bindings[name].data for name in self.output_names]

    def preprocess(self, image_path: Union[str, Path]) -> torch.Tensor:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Read image failed: {image_path}.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (H,W,C)→(C,H,W)
        image = np.expand_dims(image, axis=0)  # (C,H,W)→(1,C,H,W)

        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        dtype_map = {
            np.dtype("float32"): torch.float32,
            np.dtype("float16"): torch.float16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
        }
        expected_dtype = dtype_map.get(self.bindings[self.main_input_name].dtype, tensor.dtype)
        return tensor.type(expected_dtype)

    @staticmethod
    def postprocess(x: torch.Tensor) -> np.ndarray:
        x = x.squeeze(0).cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        x = (x * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    def process_image(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
        input_tensor = self.preprocess(input_path)
        output_tensor = self.inference(input_tensor)
        output_image = self.postprocess(output_tensor)
        cv2.imwrite(str(output_path), output_image)
        LOGGER.info(f"Image saved to '{output_path}'.")
