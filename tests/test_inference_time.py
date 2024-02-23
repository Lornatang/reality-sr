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
# ==============================================================================import torch
import torch
from torch import Tensor
from torch import nn

from reality_sr.models.edsrnet import edsrnet_x2, edsrnet_x3, edsrnet_x4, edsrnet_x8
from reality_sr.models.rfdb import rfdnet_x4
from reality_sr.models.rrdbnet import rrdbnet_x2, rrdbnet_x3, rrdbnet_x4, rrdbnet_x8

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main():
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu")
    tensor_shape = [1, 3, 256, 256]
    cuda_tensor = torch.randn(tensor_shape).to(cuda_device)
    cpu_tensor = cuda_tensor.to(cpu_device)

    print(f"=============== CUDA ===============")
    benchmark_all_edsr_models(cuda_device, cuda_tensor)
    benchmark_all_rfdb_models(cuda_device, cuda_tensor)
    benchmark_all_rrdb_models(cuda_device, cuda_tensor)

    print(f"=============== CPU ===============")
    benchmark_all_edsr_models(cpu_device, cpu_tensor)
    benchmark_all_rfdb_models(cpu_device, cpu_tensor)
    benchmark_all_rrdb_models(cpu_device, cpu_tensor)


def build_all_edsr_model(tensor: Tensor, device: torch.device):
    x2_model = edsrnet_x2()
    x3_model = edsrnet_x3()
    x4_model = edsrnet_x4()
    x8_model = edsrnet_x8()

    x2_model = x2_model.to(device)
    x3_model = x3_model.to(device)
    x4_model = x4_model.to(device)
    x8_model = x8_model.to(device)

    _ = x2_model(tensor)
    _ = x3_model(tensor)
    _ = x4_model(tensor)
    _ = x8_model(tensor)

    return x2_model, x3_model, x4_model, x8_model


def build_all_rfdb_model(tensor: Tensor, device: torch.device):
    x4_model = rfdnet_x4()

    x4_model = x4_model.to(device)

    _ = x4_model(tensor)

    return x4_model


def build_all_rrdb_model(tensor: Tensor, device: torch.device):
    x2_model = rrdbnet_x2()
    x3_model = rrdbnet_x3()
    x4_model = rrdbnet_x4()
    x8_model = rrdbnet_x8()

    x2_model = x2_model.to(device)
    x3_model = x3_model.to(device)
    x4_model = x4_model.to(device)
    x8_model = x8_model.to(device)

    _ = x2_model(tensor)
    _ = x3_model(tensor)
    _ = x4_model(tensor)
    _ = x8_model(tensor)

    return x2_model, x3_model, x4_model, x8_model


def benchmark_model(tensor: Tensor, model: nn.Module, iterations: int = 50) -> (float, float):
    times = torch.zeros(iterations)
    with torch.no_grad():
        for current_iter in range(iterations):
            starter.record()
            _ = model(tensor)
            ender.record()
            if tensor.device.type != "cpu":
                torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times[current_iter] = curr_time

    if tensor.device.type != "cpu":
        torch.cuda.empty_cache()

    mean_time = times.mean().item()
    return mean_time, 1000 / mean_time


def benchmark_all_edsr_models(device, tensor):
    all_edsr_models = build_all_edsr_model(tensor, device)
    for i, model in zip([2, 3, 4, 8], all_edsr_models):
        inference_time, fps = benchmark_model(tensor, model)
        print(f"edsrnet_x{i}: {inference_time:.1f} ms, {fps:.1f} fps")


def benchmark_all_rfdb_models(device, tensor):
    all_rfdb_models = build_all_rfdb_model(tensor, device)
    inference_time, fps = benchmark_model(tensor, all_rfdb_models)
    print(f"rfdb_x4: {inference_time:.1f} ms, {fps:.1f} fps")


def benchmark_all_rrdb_models(device, tensor):
    all_rrdb_models = build_all_rrdb_model(tensor, device)
    for i, model in zip([2, 3, 4, 8], all_rrdb_models):
        inference_time, fps = benchmark_model(tensor, model)
        print(f"rrdbnet_x{i}: {inference_time:.1f} ms, {fps:.1f} fps")


if __name__ == "__main__":
    main()
