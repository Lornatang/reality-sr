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

import numpy as np
import tensorrt as trt
import torch
from alpha_dl.utils.events import LOGGER
from torch import nn


class TRTInferencer(nn.Module):
    def __init__(self, trt_path: Union[Path, str], device: torch.device = torch.device("cuda:0")) -> None:
        super().__init__()
        logger = trt.Logger(trt.Logger.WARNING)
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
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

        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(self.model.num_io_tensors):
            name = self.model.get_tensor_name(i)
            dtype = trt.nptype(self.model.get_tensor_dtype(name))
            is_input = self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                if -1 in tuple(self.model.get_tensor_shape(name)):
                    self.context.set_input_shape(name, tuple(self.model.get_tensor_profile_shape(name, 0)[1]))
            else:
                self.output_names.append(name)
            shape = tuple(self.context.get_tensor_shape(name))
            image = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, image, int(image.data_ptr()))

        self.binding_address = OrderedDict((n, data.ptr) for n, data in self.bindings.items())

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        if x.shape != self.bindings["input0"].shape:
            self.context.set_input_shape("input0", x.shape)
            self.bindings["input0"] = self.bindings["input0"]._replace(shape=x.shape)
            for name in self.output_names:
                self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
        shape = self.bindings["input0"].shape
        assert x.shape == shape, f"input size {x.shape} > max model size {shape}"
        self.binding_address["input0"] = int(x.data_ptr())
        self.context.execute_v2(list(self.binding_address.values()))
        if len(self.output_names) == 1:
            return self.bindings[self.output_names[0]].data
        else:
            return [self.bindings[x].data for x in sorted(self.output_names)]

