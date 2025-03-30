# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.transforms import functional
from augly.image import functional as aug_functional
import random
import kornia.augmentation as K

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize_rgb = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
normalize_yuv = transforms.Normalize(mean=[0.5, 0, 0], std=[0.5, 1, 1])
unnormalize_yuv = transforms.Normalize(mean=[-0.5/0.5, 0, 0], std=[1/0.5, 1/1, 1/1])


def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def round_pixel(x):
    """ 
    Round pixel values to nearest integer. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = torch.round(x_pixel).clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def clamp_pixel(x):
    """ 
    Clamp pixel values to 0 255. 
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    x_pixel = 255 * unnormalize_img(x)
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def project_linf(x, y, radius):
    """ 
    Clamp x so that Linf(x,y)<=radius
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        radius: Radius of Linf ball for the images in pixel space [0, 255]
     """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = torch.clamp(delta, -radius, radius)
    delta = (delta / 255.0) / image_std.to(x.device)
    return y + delta

def psnr(x, y):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]

    # left = int(x.size[0]/2-new_edges_size[0]/2)
    # upper = int(x.size[1]/2-new_edges_size[1]/2)
    # right = left + new_edges_size[0]
    # lower = upper + new_edges_size[1]

    # return x.crop((left, upper, right, lower))
    x = functional.center_crop(x, new_edges_size)
    return x

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust constrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

def gaussian_blur(x, sigma=1):
    """ Add gaussian blur to image
    Args:
        x: Tensor image
        sigma: sigma of gaussian kernel
    """
    x = unnormalize_img(x)
    x = functional.gaussian_blur(x, sigma=sigma, kernel_size=21)
    x = normalize_img(x)
    return x

def random_mask_2d(images, mask_size=32):
    """
    images: (B, C, H, W) tensor
    mask_size: integer, e.g. 32

    Returns masked_images where only one random 32x32 patch 
    remains (non-zero) per sample. The rest is zeroed out.
    """
    B, C, H, W = images.shape
    if H < mask_size or W < mask_size:
        raise ValueError(
            f"Image size ({H}x{W}) is smaller than mask size ({mask_size}x{mask_size})."
        )

    # Create an all-zeros mask the same size as 'images'
    mask = torch.zeros_like(images)

    # For each sample in the batch, randomly select a 32x32 region
    for b in range(B):
        y = random.randint(0, H - mask_size)
        x = random.randint(0, W - mask_size)
        # Set that region to 1 in the mask
        mask[b, :, y : y + mask_size, x : x + mask_size] = 1.0

    # Elementwise-multiply images by the mask
    masked_images = images * mask
    return masked_images

def random_grid_mask_2d(images, mask_size=32):
    """
    images: (B, C, H, W) tensor
    mask_size: integer, e.g. 32

    Returns masked_images where only one random tile from fixed grid positions 
    remains (non-zero) per sample. The rest is zeroed out.
    """
    B, C, H, W = images.shape
    if H < mask_size or W < mask_size:
        raise ValueError(
            f"Image size ({H}x{W}) is smaller than mask size ({mask_size}x{mask_size})."
        )

    # Calculate number of tiles in height and width
    num_tiles_h = H // mask_size  # Integer division for fixed grid
    num_tiles_w = W // mask_size

    if num_tiles_h == 0 or num_tiles_w == 0:
        raise ValueError(
            f"Image dimensions ({H}x{W}) must support at least one {mask_size}x{mask_size} tile."
        )

    # Create an all-zeros mask the same size as 'images'
    mask = torch.zeros_like(images)

    # For each sample in the batch, randomly select a tile from the fixed grid
    for b in range(B):
        # Randomly choose a tile index
        tile_y = random.randint(0, num_tiles_h - 1)
        tile_x = random.randint(0, num_tiles_w - 1)
        
        # Convert tile indices to pixel coordinates
        y = tile_y * mask_size
        x = tile_x * mask_size
        
        # Set that fixed tile region to 1 in the mask
        mask[b, :, y:y + mask_size, x:x + mask_size] = 1.0

    # Elementwise-multiply images by the mask
    masked_images = images * mask
    return masked_images

def fixed_grid_mask_2d(images, mask_size=32, tile_y=0, tile_x=0):
    """
    images: (B, C, H, W) tensor
    mask_size: integer, e.g. 32
    tile_y: integer, fixed tile index in height (default 0)
    tile_x: integer, fixed tile index in width (default 0)

    Returns masked_images where only the specified fixed tile 
    remains (non-zero) per sample. The rest is zeroed out.
    """
    B, C, H, W = images.shape
    if H < mask_size or W < mask_size:
        raise ValueError(
            f"Image size ({H}x{W}) is smaller than mask size ({mask_size}x{mask_size})."
        )

    # Calculate number of tiles in height and width
    num_tiles_h = H // mask_size  # Integer division for fixed grid
    num_tiles_w = W // mask_size

    if num_tiles_h == 0 or num_tiles_w == 0:
        raise ValueError(
            f"Image dimensions ({H}x{W}) must support at least one {mask_size}x{mask_size} tile."
        )

    # Ensure the specified tile indices are valid
    if tile_y < 0 or tile_y >= num_tiles_h:
        raise ValueError(f"tile_y ({tile_y}) must be between 0 and {num_tiles_h - 1}.")
    if tile_x < 0 or tile_x >= num_tiles_w:
        raise ValueError(f"tile_x ({tile_x}) must be between 0 and {num_tiles_w - 1}.")

    # Create an all-zeros mask the same size as 'images'
    mask = torch.zeros_like(images)

    # Convert fixed tile indices to pixel coordinates
    y = tile_y * mask_size
    x = tile_x * mask_size

    # Set the fixed tile region to 1 in the mask for all samples
    mask[:, :, y:y + mask_size, x:x + mask_size] = 1.0

    # Elementwise-multiply images by the mask
    masked_images = images * mask
    return masked_images