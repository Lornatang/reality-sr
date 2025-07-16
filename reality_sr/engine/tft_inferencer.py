# Copyright (c) AlphaBetter. All rights reserved.
import json
from pathlib import Path
from typing import Union

import numpy as np
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
import tensorrt as trt
import torch
from reality_sr.utils.events import LOGGER
from torch import nn


class TRTInferencer(nn.Module):
    def __init__(self, trt_path: Union[Path, str], device: torch.device = torch.device("cuda:0")) -> None:
        super().__init__()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(trt_path), "rb") as file, trt.Runtime(logger) as runtime:
            try:
                metadata_length = int.from_bytes(file.read(4), byteorder="little")
                metadata = json.loads(file.read(metadata_length).decode("utf-8"))
                dla = metadata.get("dla", None)
                if dla is not None:
                    runtime.DLA_core = int(dla)
            except UnicodeDecodeError:
                file.seek(0)
            self.model = runtime.deserialize_cuda_engine(file.read())

        # Model context.
        try:
            self.context = self.model.create_execution_context()
        except Exception as e:  # model is None.
            LOGGER.error(f"TensorRT model exported with a different version than {trt.__version__}\n")
            raise e

        self.device = device
        if self.device.type == "cpu":
            self.device = torch.device("cuda:0")

        self.bindings = []
        self.io_tensors = {}
        for index in range(self.model.num_io_tensors):
            name = self.model.get_tensor_name(index)
            dtype = trt.nptype(self.model.get_tensor_dtype(name))
            is_input = self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                if -1 in tuple(self.model.get_tensor_shape(name)):
                    self.context.set_input_shape(name, tuple(self.model.get_tensor_profile_shape(name, 0)[1]))
            shape = tuple(self.context.get_tensor_shape(name))
            image = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.io_tensors[name] = image
            self.bindings.append(int(image.data_ptr()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).type(self.io_tensors["input0"].dtype)
        self.io_tensors["input0"].copy_(x)
        self.context.execute_v2(bindings=self.bindings)
        return self.io_tensors["output0"]
