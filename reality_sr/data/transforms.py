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
import torchvision.transforms.functional as F_vision
from torch import Tensor

__all__ = [
    "center_crop", "random_crop", "random_rotate", "random_horizontally_flip", "random_vertically_flip", "center_crop_torch", "random_crop_torch",
    "random_rotate_torch", "random_horizontally_flip_torch", "random_vertically_flip_torch"
]


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    r"""Intercept the specified center area of the image

    Args:
        image (np.ndarray): image read by OpenCV library
        image_size (int): The size of the intercepted image

    Returns:
        patch_image (np.ndarray): the intercepted image
    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # screenshot
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    r"""Randomly intercept the specified area of the image

    Args:
        image (np.ndarray): image read by OpenCV library
        image_size (int): The size of the intercepted image

    Returns:
        patch_image (np.ndarray): the intercepted image
    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # screenshot
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_rotate(image,
                  angles: list,
                  center: tuple[int, int] = None,
                  scale_factor: float = 1.0) -> np.ndarray:
    r"""Randomly rotate the image

    Args:
        image (np.ndarray): image read by OpenCV library
        angles (list): list of random rotation angles
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): Rotation scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): rotated image

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    r"""Randomly flip the image left and right

    Args:
        image (np.ndarray): image read by OpenCV library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): flipped image
    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    r"""Randomly flip the image up and down

    Args:
        image (np.ndarray): image read by OpenCV library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): flipped image
    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image


def center_crop_torch(
        gt_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        lr_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        gt_patch_size: int,
        upscale_factor: int,
) -> [np.ndarray, np.ndarray] or [Tensor, Tensor] or [list[np.ndarray], list[np.ndarray]] or [list[Tensor], list[Tensor]]:
    r"""Intercept two images to specify the center area

    Args:
        gt_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (np.ndarray or Tensor or): the intercepted ground truth image
        lr_images (np.ndarray or Tensor or): low-resolution intercepted images
    """
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = (lr_image_height - lr_patch_size) // 2
    lr_left = (lr_image_width - lr_patch_size) // 2

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_crop_torch(
        gt_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        lr_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        gt_patch_size: int,
        upscale_factor: int,
) -> [np.ndarray, np.ndarray] or [Tensor, Tensor] or [list[np.ndarray], list[np.ndarray]] or [list[Tensor], list[Tensor]]:
    r"""Randomly intercept two images in the specified area

    Args:
        gt_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): ground truth images read by PyTorch
        lr_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): Low resolution images read by PyTorch
        gt_patch_size (int): the size of the ground truth image after interception
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size

    Returns:
        gt_images (np.ndarray or Tensor or): the intercepted ground truth image
        lr_images (np.ndarray or Tensor or): low-resolution intercepted images
    """
    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Calculate the size of the low-resolution image that needs to be intercepted
    lr_patch_size = gt_patch_size // upscale_factor

    # Just need to find the top and left coordinates of the image
    lr_top = random.randint(0, lr_image_height - lr_patch_size)
    lr_left = random.randint(0, lr_image_width - lr_patch_size)

    # Capture low-resolution images
    if input_type == "Tensor":
        lr_images = [lr_image[
                     :,
                     :,
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size] for lr_image in lr_images]
    else:
        lr_images = [lr_image[
                     lr_top: lr_top + lr_patch_size,
                     lr_left: lr_left + lr_patch_size,
                     ...] for lr_image in lr_images]

    # Intercept the ground truth image
    gt_top, gt_left = int(lr_top * upscale_factor), int(lr_left * upscale_factor)

    if input_type == "Tensor":
        gt_images = [v[
                     :,
                     :,
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size] for v in gt_images]
    else:
        gt_images = [v[
                     gt_top: gt_top + gt_patch_size,
                     gt_left: gt_left + gt_patch_size,
                     ...] for v in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_rotate_torch(
        gt_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        lr_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        upscale_factor: int,
        angles: list,
        gt_center: tuple = None,
        lr_center: tuple = None,
        rotate_scale_factor: float = 1.0
) -> [np.ndarray, np.ndarray] or [Tensor, Tensor] or [list[np.ndarray], list[np.ndarray]] or [list[Tensor], list[Tensor]]:
    r"""Randomly rotate the image

    Args:
        gt_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): ground truth images read by the PyTorch library
        lr_images (np.ndarray | Tensor | list[np.ndarray] | list[Tensor]): low-resolution images read by the PyTorch library
        angles (list): list of random rotation angles
        upscale_factor (int): the ground truth image size is a magnification of the low resolution image size
        gt_center (optional, tuple[int, int]): The center point of the ground truth image selection. Default: ``None``
        lr_center (optional, tuple[int, int]): Low resolution image selection center point. Default: ``None``
        rotate_scale_factor (optional, float): Rotation scaling factor. Default: 1.0

    Returns:
        gt_images (np.ndarray or Tensor or): ground truth image after rotation
        lr_images (np.ndarray or Tensor or): Rotated low-resolution images
    """
    # Randomly choose the rotation angle
    angle = random.choice(angles)

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if input_type == "Tensor":
        lr_image_height, lr_image_width = lr_images[0].size()[-2:]
    else:
        lr_image_height, lr_image_width = lr_images[0].shape[0:2]

    # Rotate the low-res image
    if lr_center is None:
        lr_center = [lr_image_width // 2, lr_image_height // 2]

    lr_matrix = cv2.getRotationMatrix2D(lr_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        lr_images = [F_vision.rotate(lr_image, angle, center=lr_center) for lr_image in lr_images]
    else:
        lr_images = [cv2.warpAffine(lr_image, lr_matrix, (lr_image_width, lr_image_height)) for lr_image in lr_images]

    # rotate the ground truth image
    gt_image_width = int(lr_image_width * upscale_factor)
    gt_image_height = int(lr_image_height * upscale_factor)

    if gt_center is None:
        gt_center = [gt_image_width // 2, gt_image_height // 2]

    gt_matrix = cv2.getRotationMatrix2D(gt_center, angle, rotate_scale_factor)

    if input_type == "Tensor":
        gt_images = [F_vision.rotate(gt_image, angle, center=gt_center) for gt_image in gt_images]
    else:
        gt_images = [cv2.warpAffine(gt_image, gt_matrix, (gt_image_width, gt_image_height)) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_horizontally_flip_torch(
        gt_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        lr_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        p: float = 0.5
) -> [np.ndarray, np.ndarray] or [Tensor, Tensor] or [list[np.ndarray], list[np.ndarray]] or [list[Tensor], list[Tensor]]:
    r"""Randomly flip the image up and down

    Args:
        gt_images (np.ndarray): ground truth images read by the PyTorch library
        lr_images (np.ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        gt_images (np.ndarray or Tensor or): flipped ground truth images
        lr_images (np.ndarray or Tensor or): flipped low-resolution images
    """
    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.hflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.hflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 1) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 1) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images


def random_vertically_flip_torch(
        gt_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        lr_images: np.ndarray | Tensor | list[np.ndarray] | list[Tensor],
        p: float = 0.5
) -> [np.ndarray, np.ndarray] or [Tensor, Tensor] or [list[np.ndarray], list[np.ndarray]] or [list[Tensor], list[Tensor]]:
    r"""Randomly flip the image left and right

    Args:
        gt_images (np.ndarray): ground truth images read by the PyTorch library
        lr_images (np.ndarray): low resolution images read by the PyTorch library
        p (optional, float): flip probability. Default: 0.5

    Returns:
        gt_images (np.ndarray or Tensor or): flipped ground truth images
        lr_images (np.ndarray or Tensor or): flipped low-resolution images
    """
    # Randomly generate flip probability
    flip_prob = random.random()

    if not isinstance(gt_images, list):
        gt_images = [gt_images]
    if not isinstance(lr_images, list):
        lr_images = [lr_images]

    # detect input image type
    input_type = "Tensor" if torch.is_tensor(lr_images[0]) else "Numpy"

    if flip_prob > p:
        if input_type == "Tensor":
            lr_images = [F_vision.vflip(lr_image) for lr_image in lr_images]
            gt_images = [F_vision.vflip(gt_image) for gt_image in gt_images]
        else:
            lr_images = [cv2.flip(lr_image, 0) for lr_image in lr_images]
            gt_images = [cv2.flip(gt_image, 0) for gt_image in gt_images]

    # When the input has only one image
    if len(gt_images) == 1:
        gt_images = gt_images[0]
    if len(lr_images) == 1:
        lr_images = lr_images[0]

    return gt_images, lr_images
