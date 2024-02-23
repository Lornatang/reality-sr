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
import random

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from scipy import special
from scipy.stats import multivariate_normal
from torch import Tensor, nn
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_vision

from reality_sr.utils.imgproc import filter2D_torch

__all__ = [
    "degradation_process", "random_mixed_kernels", "random_add_gaussian_noise_torch", "random_add_poisson_noise_torch", "generate_sinc_kernel",
]


def degradation_process(
        gt: Tensor,
        gaussian_kernel1: Tensor,
        gaussian_kernel2: Tensor,
        sinc_kernel: Tensor,
        upscale_factor: int,
        degradation_process_parameters_dict: DictConfig,
        jpeg_operation: nn.Module = None,
        usm_sharpener: nn.Module = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""degradation processing

    Args:
        gt (Tensor): the input ground truth image
        gaussian_kernel1 (Tensor): Gaussian kernel used for the first degradation
        gaussian_kernel2 (Tensor): The Gaussian kernel used for the second degradation
        sinc_kernel (Tensor): Sinc kernel used for degradation
        upscale_factor (int): zoom factor
        degradation_process_parameters_dict (DictConfig): A dictionary containing degradation processing parameters
        jpeg_operation (nn.Module): JPEG compression model. Default: ``None``
        usm_sharpener (nn.Module): USM sharpening model. Default: ``None``

    Returns:
        gt_usm (Tensor): GT image after USM processing
        gt (Tensor): GT image
        lr (Tensor): Low-resolution image after degradation processing

    """
    # Get the ground truth image size
    image_height, image_width = gt.size()[2:4]

    # When the sharpening operation is not suitable, the GT after sharpening is equal to GT
    gt_usm = gt

    # Sharpen the ground truth image
    if usm_sharpener is not None:
        gt_usm = usm_sharpener(gt)

    # The first degradation processing
    # Gaussian
    if np.random.uniform() <= degradation_process_parameters_dict.FIRST_BLUR_PROBABILITY:
        out = filter2D_torch(gt_usm, gaussian_kernel1)

    # Resize
    updown_type = random.choices(["up", "down", "keep"], OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_PROBABILITY1))[0]
    if updown_type == "up":
        scale = np.random.uniform(1, OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_RANGE1)[1])
    elif updown_type == "down":
        scale = np.random.uniform(OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_RANGE1)[0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F_torch.interpolate(out, scale_factor=scale, mode=mode)

    # noise
    if np.random.uniform() < degradation_process_parameters_dict.GAUSSIAN_NOISE_PROBABILITY1:
        out = random_add_gaussian_noise_torch(
            image=out,
            sigma_range=tuple(degradation_process_parameters_dict.NOISE_RANGE1),
            clip=True,
            rounds=False,
            gray_prob=degradation_process_parameters_dict.GRAY_NOISE_PROBABILITY1)
    else:
        out = random_add_poisson_noise_torch(
            image=out,
            scale_range=tuple(degradation_process_parameters_dict.POISSON_SCALE_RANGE1),
            gray_prob=degradation_process_parameters_dict.GRAY_NOISE_PROBABILITY1,
            clip=True,
            rounds=False)

    # JPEG compression
    quality = out.new_zeros(out.size(0))
    quality = quality.uniform_(*OmegaConf.to_container(degradation_process_parameters_dict.JPEG_RANGE1))
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeg_operation(out, quality)

    # second degradation processing
    # Gaussian
    if np.random.uniform() < degradation_process_parameters_dict.SECOND_BLUR_PROBABILITY:
        out = filter2D_torch(out, gaussian_kernel2)

    # Resize
    updown_type = random.choices(["up", "down", "keep"], OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_PROBABILITY2))[0]
    if updown_type == "up":
        scale = np.random.uniform(1, OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_RANGE2)[1])
    elif updown_type == "down":
        scale = np.random.uniform(OmegaConf.to_container(degradation_process_parameters_dict.RESIZE_RANGE2)[0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F_torch.interpolate(out,
                              size=(int(image_height / upscale_factor * scale),
                                    int(image_width / upscale_factor * scale)),
                              mode=mode)

    # Noise
    if np.random.uniform() < degradation_process_parameters_dict.GAUSSIAN_NOISE_PROBABILITY2:
        out = random_add_gaussian_noise_torch(
            image=out,
            sigma_range=tuple(degradation_process_parameters_dict.NOISE_RANGE2),
            clip=True,
            rounds=False,
            gray_prob=degradation_process_parameters_dict.GRAY_NOISE_PROBABILITY2)
    else:
        out = random_add_poisson_noise_torch(
            image=out,
            scale_range=tuple(degradation_process_parameters_dict.POISSON_SCALE_RANGE2),
            gray_prob=degradation_process_parameters_dict.GRAY_NOISE_PROBABILITY2,
            clip=True,
            rounds=False)

    if np.random.uniform() < 0.5:
        # Zoom out -> Sinc filter -> JPEG compression
        out = F_torch.interpolate(out,
                                  size=(image_height // upscale_factor, image_width // upscale_factor),
                                  mode=random.choice(["area", "bilinear", "bicubic"]))

        out = filter2D_torch(out, sinc_kernel)

        quality = out.new_zeros(out.size(0))
        quality = quality.uniform_(*OmegaConf.to_container(degradation_process_parameters_dict.JPEG_RANGE2))
        out = torch.clamp(out, 0, 1)
        out = jpeg_operation(out, quality)
    else:
        # JPEG compression -> reduction -> Sinc filter
        quality = out.new_zeros(out.size(0))
        quality = quality.uniform_(*OmegaConf.to_container(degradation_process_parameters_dict.JPEG_RANGE2))
        out = torch.clamp(out, 0, 1)
        out = jpeg_operation(out, quality)

        out = F_torch.interpolate(out,
                                  size=(image_height // upscale_factor, image_width // upscale_factor),
                                  mode=random.choice(["area", "bilinear", "bicubic"]))

        out = filter2D_torch(out, sinc_kernel)

    # Intercept the pixel range of the output image
    lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    return gt_usm, gt, lr


def random_mixed_kernels(kernel_type: list,
                         kernel_prob: list,
                         kernel_size: int,
                         sigma_x_range: list,
                         sigma_y_range: list,
                         rotation_range: list,
                         generalized_kernel_beta_range: list,
                         plateau_kernel_beta_range: list,
                         noise_range: tuple = None) -> np.ndarray:
    """Randomly generate mixed kernels

    Args:
        kernel_type (list): a list name of gaussian kernel types
        kernel_prob (list): corresponding kernel probability for each kernel type
        kernel_size (int): Gaussian kernel size
        sigma_x_range (list): Sigma range along the horizontal axis
        sigma_y_range (list): Sigma range along the vertical axis
        rotation_range (list): Gaussian kernel rotation matrix angle range value
        generalized_kernel_beta_range (list): Gaussian kernel beta matrix angle range value
        plateau_kernel_beta_range (list): Plateau Gaussian Kernel Beta Matrix Angle Range Values
        noise_range (optional, tuple): multiplicative kernel noise, [0.75, 1.25]. Default: ``None``

    Returns:
        mixed kernels (np.ndarray): Mixed kernels

    """
    kernel_type = random.choices(kernel_type, kernel_prob)[0]
    if kernel_type == "isotropic":
        mixed_kernels = _random_bivariate_gaussian_kernel(kernel_size,
                                                          sigma_x_range,
                                                          sigma_y_range,
                                                          rotation_range,
                                                          noise_range,
                                                          True)
    elif kernel_type == "anisotropic":
        mixed_kernels = _random_bivariate_gaussian_kernel(kernel_size,
                                                          sigma_x_range,
                                                          sigma_y_range,
                                                          rotation_range,
                                                          noise_range,
                                                          False)
    elif kernel_type == "generalized_isotropic":
        mixed_kernels = _random_bivariate_generalized_gaussian_kernel(kernel_size,
                                                                      sigma_x_range,
                                                                      sigma_y_range,
                                                                      rotation_range,
                                                                      generalized_kernel_beta_range,
                                                                      noise_range,
                                                                      True)
    elif kernel_type == "generalized_anisotropic":
        mixed_kernels = _random_bivariate_generalized_gaussian_kernel(kernel_size,
                                                                      sigma_x_range,
                                                                      sigma_y_range,
                                                                      rotation_range,
                                                                      generalized_kernel_beta_range,
                                                                      noise_range,
                                                                      False)
    elif kernel_type == "plateau_isotropic":
        mixed_kernels = _random_bivariate_plateau_gaussian_kernel(kernel_size,
                                                                  sigma_x_range,
                                                                  sigma_y_range,
                                                                  rotation_range,
                                                                  plateau_kernel_beta_range,
                                                                  None,
                                                                  True)
    elif kernel_type == "plateau_anisotropic":
        mixed_kernels = _random_bivariate_plateau_gaussian_kernel(kernel_size,
                                                                  sigma_x_range,
                                                                  sigma_y_range,
                                                                  rotation_range,
                                                                  plateau_kernel_beta_range,
                                                                  None,
                                                                  False)
    else:
        mixed_kernels = _random_bivariate_gaussian_kernel(kernel_size,
                                                          sigma_x_range,
                                                          sigma_y_range,
                                                          rotation_range,
                                                          noise_range,
                                                          True)

    return mixed_kernels


def random_add_gaussian_noise_torch(image: Tensor,
                                    sigma_range: tuple = (0, 1.0),
                                    gray_prob: int = 0,
                                    clip: bool = True,
                                    rounds: bool = False) -> Tensor:
    """Random add gaussian noise to image (PyTorch)

    Args:
        image (Tensor): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (Tensor): Add gaussian noise to image

    """
    noise = _random_generate_gaussian_noise_torch(image, sigma_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def random_add_poisson_noise_torch(image: Tensor,
                                   scale_range: tuple = (0, 1.0),
                                   gray_prob: int = 0,
                                   clip: bool = True,
                                   rounds: bool = False) -> Tensor:
    """Random add gaussian noise to image (PyTorch)

    Args:
        image (Tensor): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (Tensor): Add poisson noise to image

    """
    noise = _random_generate_poisson_noise_torch(image, scale_range, gray_prob)
    out = image + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def generate_sinc_kernel(cutoff: float, kernel_size: int, padding: int = 0) -> np.ndarray:
    """2D sinc filter

    Args:
        cutoff (float): Cutoff frequency in radians (pi is max)
        kernel_size (int): Horizontal and vertical size, must be odd
        padding (optional, int): Pad kernel size to desired size, must be odd or zero. Default: 0

    Returns:
        sinc_kernel (np.ndarray): Sinc kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."

    np.seterr(divide="ignore", invalid="ignore")

    sinc_kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)) / (2 * np.pi * np.sqrt(
            (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)), [kernel_size, kernel_size])
    sinc_kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    sinc_kernel = sinc_kernel / np.sum(sinc_kernel)

    if padding > kernel_size:
        pad_size = (padding - kernel_size) // 2
        sinc_kernel = np.pad(sinc_kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    return sinc_kernel


def _mesh_grid(kernel_size: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields
    over N-D grids, given one-dimensional coordinate arrays x1, x2,..., xn.

    Args:
        kernel_size (int): Gaussian kernel size

    Returns:
        xy (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size, 2)
        xx (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size)
        yy (np.ndarray): Gaussian kernel with this scale ->(kernel_size, kernel_size)

    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)),
                    yy.reshape(kernel_size * kernel_size, 1))).reshape(kernel_size, kernel_size, 2)

    return xy, xx, yy


def _calculate_rotate_sigma_matrix(sigma_x: float, sigma_y: float, theta: float) -> np.ndarray:
    """Calculate rotated sigma matrix

    Args:
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement

    Returns:
        out (np.ndarray): Rotated sigma matrix

    """
    d_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    out = np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

    return out


def _calculate_probability_density(sigma_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Calculate probability density function of the bivariate Gaussian distribution

    Args:
        sigma_matrix (np.ndarray): with the shape (2, 2)
        grid (np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        probability_density (np.ndarray): un-normalized kernel

    """
    inverse_sigma = np.linalg.inv(sigma_matrix)
    probability_density = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))

    return probability_density


def _calculate_cumulative_density(skew_matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Calculate the CDF of the standard bivariate Gaussian distribution.
    Used in skewed Gaussian distribution

    Args:
        skew_matrix (np.ndarray): skew matrix
        grid (np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size

    Returns:
        cumulative_density (np.ndarray): Cumulative density function

    """
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    grid = np.dot(grid, skew_matrix)
    cumulative_density = rv.cdf(grid)

    return cumulative_density


def _generate_bivariate_gaussian_kernel(kernel_size: int,
                                        sigma_x: float,
                                        sigma_y: float, theta: float,
                                        grid: np.ndarray = None,
                                        isotropic: bool = True) -> np.ndarray:
    """Generate a bivariate isotropic or anisotropic Gaussian kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value in the horizontal axis direction
        sigma_y (float): sigma value along the vertical axis
        theta (float): Radian measurement
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: ``None``
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. Default: ``True``

    Returns:
        bivariate_gaussian_kernel (np.ndarray): Bivariate gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    bivariate_gaussian_kernel = _calculate_probability_density(sigma_matrix, grid)
    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


def _generate_bivariate_generalized_gaussian_kernel(kernel_size: int,
                                                    sigma_x: float,
                                                    sigma_y: float,
                                                    theta: float,
                                                    beta: float,
                                                    grid: np.ndarray = None,
                                                    isotropic: bool = True) -> np.ndarray:
    """Generate a bivariate generalized Gaussian kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): bivariate generalized gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_generalized_gaussian_kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta))
    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel)

    return bivariate_generalized_gaussian_kernel


def _generate_bivariate_plateau_gaussian_kernel(kernel_size: int,
                                                sigma_x: float,
                                                sigma_y: float,
                                                theta: float,
                                                beta: float,
                                                grid: np.ndarray = None,
                                                isotropic: bool = True) -> np.ndarray:
    """Generate a plateau-like anisotropic kernel

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x (float): Sigma value of the horizontal axis
        sigma_y (float): Sigma value of the vertical axis
        theta (float): Radian measurement
        beta (float): shape parameter, beta = 1 is the normal distribution
        grid (optional, np.ndarray): generated by :func:`mesh_grid`, with the shape (K, K, 2), K is the kernel size. Default: None
        isotropic (optional, bool): Set to `True` for homosexual Gaussian kernel, set to `False` for heterosexual Gaussian kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (np.ndarray): bivariate plateau gaussian kernel

    """
    if grid is None:
        grid, _, _ = _mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sigma_x ** 2, 0], [0, sigma_x ** 2]])
    else:
        sigma_matrix = _calculate_rotate_sigma_matrix(sigma_x, sigma_y, theta)

    inverse_sigma = np.linalg.inv(sigma_matrix)
    bivariate_plateau_gaussian_kernel = np.reciprocal(np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1)
    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


def _random_bivariate_gaussian_kernel(kernel_size: int,
                                      sigma_x_range: list,
                                      sigma_y_range: list,
                                      rotation_range: list,
                                      noise_range: list = None,
                                      isotropic: bool = True) -> np.ndarray:
    """Randomly generate bivariate isotropic or anisotropic Gaussian kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (list): Sigma range along the horizontal axis
        sigma_y_range (list): Sigma range along the vertical axis
        rotation_range (list): Gaussian kernel rotation matrix angle range value
        noise_range(optional, tuple): multiplicative kernel noise. Default: None
        isotropic (optional, bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_gaussian_kernel (np.ndarray): Bivariate gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
        assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    bivariate_gaussian_kernel = _generate_bivariate_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation,
                                                                    isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_gaussian_kernel.shape)
        bivariate_gaussian_kernel = bivariate_gaussian_kernel * noise

    bivariate_gaussian_kernel = bivariate_gaussian_kernel / np.sum(bivariate_gaussian_kernel)

    return bivariate_gaussian_kernel


def _random_bivariate_generalized_gaussian_kernel(kernel_size: int,
                                                  sigma_x_range: list,
                                                  sigma_y_range: list,
                                                  rotation_range: list,
                                                  beta_range: list,
                                                  noise_range: list = None,
                                                  isotropic: bool = True) -> np.ndarray:
    """Randomly generate bivariate generalized Gaussian kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (list): Sigma range along the horizontal axis
        sigma_y_range (list): Sigma range along the vertical axis
        rotation_range (list): Gaussian kernel rotation matrix angle range value
        beta_range (list): Gaussian kernel beta matrix angle range value
        noise_range(optional, list): multiplicative kernel noise. Default: None
        isotropic (optional, bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_generalized_gaussian_kernel (np.ndarray): Bivariate generalized gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_generalized_gaussian_kernel = _generate_bivariate_generalized_gaussian_kernel(kernel_size,
                                                                                            sigma_x,
                                                                                            sigma_y,
                                                                                            rotation,
                                                                                            beta,
                                                                                            isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_generalized_gaussian_kernel.shape)
        bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel * noise

    bivariate_generalized_gaussian_kernel = bivariate_generalized_gaussian_kernel / np.sum(
        bivariate_generalized_gaussian_kernel)

    return bivariate_generalized_gaussian_kernel


def _random_bivariate_plateau_gaussian_kernel(kernel_size: int,
                                              sigma_x_range: list,
                                              sigma_y_range: list,
                                              rotation_range: list,
                                              beta_range: list,
                                              noise_range: list = None,
                                              isotropic: bool = True) -> np.ndarray:
    """Randomly generate bivariate plateau kernels

    Args:
        kernel_size (int): Gaussian kernel size
        sigma_x_range (list): Sigma range along the horizontal axis
        sigma_y_range (list): Sigma range along the vertical axis
        rotation_range (list): Gaussian kernel rotation matrix angle range value
        beta_range (list): Gaussian kernel beta matrix angle range value
        noise_range(tuple, list): multiplicative kernel noise. Default: None
        isotropic (bool): Set to `True` for homosexual plateau kernel, set to `False` for heterosexual plateau kernel. (Default: ``True``)

    Returns:
        bivariate_plateau_gaussian_kernel (np.ndarray): Bivariate plateau gaussian kernel

    """
    assert kernel_size % 2 == 1, "Kernel size must be an odd number."
    assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])

    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
        assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    if np.random.uniform() < 0.5:
        beta = np.random.uniform(beta_range[0], 1)
    else:
        beta = np.random.uniform(1, beta_range[1])

    bivariate_plateau_gaussian_kernel = _generate_bivariate_plateau_gaussian_kernel(kernel_size,
                                                                                    sigma_x,
                                                                                    sigma_y,
                                                                                    rotation,
                                                                                    beta,
                                                                                    isotropic=isotropic)
    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], "Wrong noise range."
        noise = np.random.uniform(noise_range[0], noise_range[1], size=bivariate_plateau_gaussian_kernel.shape)
        bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel * noise

    bivariate_plateau_gaussian_kernel = bivariate_plateau_gaussian_kernel / np.sum(bivariate_plateau_gaussian_kernel)

    return bivariate_plateau_gaussian_kernel


def _generate_gaussian_noise(image: np.ndarray, sigma: float = 10.0, gray_noise: bool = False) -> np.ndarray:
    """Generate Gaussian noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.0
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        gaussian_noise (np.array): Gaussian noise

    """
    if gray_noise:
        gaussian_noise = np.float32(np.random.randn(*(image.shape[0:2]))) * sigma / 255.
        gaussian_noise = np.expand_dims(gaussian_noise, axis=2).repeat(3, axis=2)
    else:
        gaussian_noise = np.float32(np.random.randn(*image.shape)) * sigma / 255.

    return gaussian_noise


def _generate_poisson_noise(image: np.ndarray, scale: float = 1.0, gray_noise: bool = False) -> np.ndarray:
    """Generate poisson noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        poisson_noise (np.array): Poisson noise

    """
    if gray_noise:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Round and clip image for counting vals correctly
    image = np.clip((image * 255.0).round(), 0, 255) / 255.
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    out = np.float32(np.random.poisson(image * vals) / float(vals))
    noise = out - image

    if gray_noise:
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)

    poisson_noise = noise * scale

    return poisson_noise


def _random_generate_gaussian_noise(image: np.ndarray, sigma_range: tuple = (0, 10), gray_prob: int = 0) -> np.ndarray:
    """Random generate gaussian noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 10)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        gaussian_noise (np.ndarray): Gaussian noise

    """
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    gaussian_noise = _generate_gaussian_noise(image, sigma, gray_noise)

    return gaussian_noise


def _random_generate_poisson_noise(image: np.ndarray, scale_range: tuple = (0, 1.0), gray_prob: int = 0) -> np.ndarray:
    """Random generate poisson noise (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        poisson_noise (np.ndarray): Poisson noise

    """
    scale = np.random.uniform(scale_range[0], scale_range[1])

    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False

    poisson_noise = _generate_poisson_noise(image, scale, gray_noise)

    return poisson_noise


def _add_gaussian_noise(image: np.ndarray,
                        sigma: float = 10.0,
                        clip: bool = True,
                        rounds: bool = False,
                        gray_noise: bool = False):
    """Add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma (optional, float): Noise scale (measured in range 255). Default: 10.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        out (np.ndarray): Add gaussian noise to image

    """
    noise = _generate_gaussian_noise(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _add_poisson_noise(image: np.ndarray,
                       scale: float = 1.0,
                       clip: bool = True,
                       rounds: bool = False,
                       gray_noise: bool = False):
    """Add poisson noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, bool): Whether to add grayscale noise. Default: ``False``

    Returns:
        out (np.ndarray): Add poisson noise to image

    """
    noise = _generate_poisson_noise(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _random_add_gaussian_noise(image: np.ndarray,
                               sigma_range: tuple = (0, 1.0),
                               gray_prob: int = 0,
                               clip: bool = True,
                               rounds: bool = False) -> np.ndarray:
    """Random add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (np.ndarray): Add gaussian noise to image

    """
    noise = _random_generate_gaussian_noise(image, sigma_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _random_add_poisson_noise(image: np.ndarray,
                              scale_range: tuple = (0, 1.0),
                              gray_prob: int = 0,
                              clip: bool = True,
                              rounds: bool = False) -> np.ndarray:
    """Random add gaussian noise to image (OpenCV)

    Args:
        image (np.ndarray): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Noise rounds scale. Default: ``False``

    Returns:
        out (np.ndarray): Add poisson noise to image

    """
    noise = _random_generate_poisson_noise(image, scale_range, gray_prob)
    out = image + noise

    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _generate_gaussian_noise_torch(image: Tensor,
                                   sigma: float = 10.0,
                                   gray_noise: Tensor = 0) -> Tensor:
    """Generate Gaussian noise (PyTorch)

    Args:
        image (Tensor): Input image
        sigma (float): Noise scale (measured in range 255). Default: 10.0
        gray_noise (optional, Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        gaussian_noise (Tensor): Gaussian noise

    """
    b, _, h, w = image.size()

    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(image.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*image.size()[2:4], dtype=image.dtype, device=image.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(*image.size(), dtype=image.dtype, device=image.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise

    return noise


def _generate_poisson_noise_torch(image: Tensor,
                                  scale: float = 1.0,
                                  gray_noise: Tensor = 0) -> Tensor:
    """Generate poisson noise (PyTorch)

    Args:
        image (Tensor): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        gray_noise (optional, Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        poisson_noise (Tensor): Poisson noise

    """
    b, _, h, w = image.size()

    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = F_vision.rgb_to_grayscale(image, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    image = torch.clamp((image * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(image[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(vals)) for vals in vals_list]
    vals = image.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(image * vals) / vals
    noise = out - image

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)

    poisson_noise = noise * scale

    return poisson_noise


def _random_generate_gaussian_noise_torch(image: Tensor,
                                          sigma_range: tuple = (0, 10),
                                          gray_prob: int = 0) -> Tensor:
    """Random generate gaussian noise (PyTorch)

    Args:
        image (Tensor): Input image
        sigma_range (optional, tuple): Noise range. Default: (0, 10)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        gaussian_noise (Tensor): Gaussian noise

    """
    sigma = torch.rand(image.size(0),
                       dtype=image.dtype,
                       device=image.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    gaussian_noise = _generate_gaussian_noise_torch(image, sigma, gray_noise)

    return gaussian_noise


def _random_generate_poisson_noise_torch(image: Tensor,
                                         scale_range: tuple = (0, 1.0),
                                         gray_prob: Tensor = 0) -> Tensor:
    """Random generate poisson noise (PyTorch)

    Args:
        image (Tensor): Input image
        scale_range (optional, tuple): Noise scale range. Default: (0, 1.0)
        gray_prob (optional, int): Add grayscale noise probability. Default: 0

    Returns:
        poisson_noise (Tensor): Poisson noise

    """
    scale = torch.rand(image.size(0),
                       dtype=image.dtype,
                       device=image.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(image.size(0), dtype=image.dtype, device=image.device)
    gray_noise = (gray_noise < gray_prob).float()
    poisson_noise = _generate_poisson_noise_torch(image, scale, gray_noise)

    return poisson_noise


def _add_gaussian_noise_torch(image: Tensor,
                              sigma: float = 10.0,
                              clip: bool = True,
                              rounds: bool = False,
                              gray_noise: Tensor = 0):
    """Add gaussian noise to image (PyTorch)

    Args:
        image (Tensor): Input image
        sigma (optional, float): Noise scale (measured in range 255). Default: 10.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        out (Tensor): Add gaussian noise to image

    """
    noise = _generate_gaussian_noise_torch(image, sigma, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _add_poisson_noise_torch(image: Tensor,
                             scale: float = 1.0,
                             clip: bool = True,
                             rounds: bool = False,
                             gray_noise: Tensor = 0) -> Tensor:
    """Add poisson noise to image (PyTorch)

    Args:
        image (Tensor): Input image
        scale (optional, float): Noise scale value. Default: 1.0
        clip (optional, bool): Whether to clip image pixel. If `True`, clip image pixel to [0, 1] or [0, 255]. Default: ``True``
        rounds (optional, bool): Gaussian noise rounds scale. Default: ``False``
        gray_noise (optional, Tensor): Whether to add grayscale noise. Default: 0

    Returns:
        out (Tensor): Add poisson noise to image

    """
    noise = _generate_poisson_noise_torch(image, scale, gray_noise)
    out = image + noise

    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.

    return out


def _add_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    """Add JPEG compression (OpenCV)

    Args:
        image (np.ndarray): Input image
        quality (float): JPEG compression quality

    Returns:
        jpeg_image (np.ndarray): JPEG processed image

    """
    image = np.clip(image, 0, 1)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encode_image = cv2.imencode(".jpg", image * 255., encode_params)
    jpeg_image = np.float32(cv2.imdecode(encode_image, 1)) / 255.

    return jpeg_image


def _random_add_jpg_compression(image: np.ndarray, quality_range: tuple) -> np.ndarray:
    """Random add JPEG compression (OpenCV)

    Args:
        image (np.ndarray): Input image
        quality_range (tuple): JPEG compression quality range

    Returns:
        image (np.ndarray): JPEG processed image

    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    jpeg_image = _add_jpeg_compression(image, quality)

    return jpeg_image
