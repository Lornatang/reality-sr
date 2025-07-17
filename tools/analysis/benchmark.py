import logging
import time
from typing import Any, Tuple

import torch
from tabulate import tabulate
from tqdm import tqdm

from alpha_sr.models.rrdbnet import rrdbnet_x2

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)


def benchmark(
        model: Any,
        input_shape: Tuple[int, int, int, int],
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
        iterations: int = 10,
) -> Tuple[float, float]:
    """Measures the inference speed and throughput of a given model.

    Args:
        model (Any): The model to be tested.
        input_shape (Tuple[int, int, int, int]): Shape of the input tensor (batch_size, channels, height, width).
        device (torch.device): Device to run the inference on.
        iterations (int, optional): Number of iterations for benchmarking. Defaults to 10.

    Returns:
        Tuple[float, float]: Average inference time per batch in milliseconds, and throughput (samples per second).
    """
    input_tensor = torch.randn(input_shape, dtype=dtype, device=device)
    if dtype == torch.half:
        model = model.half()
    model.eval()

    # Warmup.
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Measure inference speed.
    start_time = time.time()
    with torch.no_grad():
        for _ in tqdm(range(iterations), desc="Testing inference speed", ncols=100):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    end_time = time.time()

    # Calculate average time and throughput.
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    throughput = (input_shape[0] * iterations) / total_time

    return avg_time_ms, throughput


def main() -> None:
    device = "cuda:0"
    device = torch.device(device)
    if device.type == "cpu" and torch.cuda.is_available():
        LOGGER.warning("CUDA is available but the device is set to CPU.")

    seed = 42
    torch.manual_seed(seed)

    batch_size = 4
    channels = 3
    height = 128
    width = 128
    input_shape = (batch_size, channels, height, width)

    dtype = torch.half

    model_list = [
        {"name": "rrdbnet_n23_x2", "model": rrdbnet_x2().to(device)},
    ]

    results = []
    for item in model_list:
        avg_time, throughput = benchmark(item["model"], input_shape, dtype, device)
        results.append([
            item["name"],
            round(avg_time / batch_size, 4),
            round(throughput, 2)
        ])
    headers = ["Model Name", "Avg Sample Time (ms)", "Throughput (sample/s)"]
    print(tabulate(results, headers=headers, tablefmt="grid", colalign=("center", "center", "center")))


if __name__ == "__main__":
    main()
