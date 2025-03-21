import os
import math
import torch

import numpy as np
import torch.nn as nn

from einops import rearrange, repeat
import  torch.fft as fft

def random_resample(args, b, c, h, w, latents, sampler):
    """
    Randomly resample.
    """
    return torch.randn((b,c,1,h,w))

def last_frame_renoise(args, b,c,h,w, latent,sampler):
    """
    Renoise the last frame.
    """
    alpha = sampler.ddim_alphas[args.num_inference_steps-1] # image -> noise
    beta = 1 - alpha
    # latent = latents[:,:,[-2]]
    new_latent = (alpha)**(0.5) * latent.clone() + (1-alpha)**(0.5) * torch.randn_like(latent)
    return new_latent

def fft_2D_resample(args, b, c, h, w, latents, sampler):
    """
    resample the latents using fft.
    """
    x = latents.clone()
    LPF = gaussian_low_pass_filter_2d(x.shape, 0.25)
    x_mixed = freq_mix_2d(x, torch.randn_like(x).to(x.device), LPF)
    return x_mixed

def last_frame_resample(args, b, c, h, w, latent, sampler):
    """
    Resample the last frame.
    """
    d = 0.1
    alpha = sampler.ddim_alphas[args.num_inference_steps-1] # image -> noise
    beta = 1 - alpha
    new_latent = (alpha)**(0.5) * latent.clone() + (1-alpha)**(0.5) * torch.randn_like(latent)
    new_noise_gaussian = sample_in_neighborhood_normal(new_latent, d)

    return new_noise_gaussian

def gaussian_low_pass_filter_2d(shape, d_s=0.25):
    """
    Compute the Gaussian low pass filter mask using vectorized operations, ensuring exact
    calculation to match the old loop-based implementation.

    Args:
        shape: shape of the filter (volume)
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
        d_t: normalized stop frequency for temporal dimension (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    if d_s == 0:
        return torch.zeros(shape)

    # Create normalized coordinate grids for T, H, W
    # Generate indices as in the old loop-based method
    h = torch.arange(H).float() * 2 / H - 1
    w = torch.arange(W).float() * 2 / W - 1
    
    # Use meshgrid to create 3D grid of coordinates
    grid_h, grid_w = torch.meshgrid(h, w, indexing='ij')

    # Compute squared distance from the center, adjusted for the frequency cut-offs
    d_square = ((grid_h * (1 / d_s)).pow(2) + (grid_w * (1 / d_s)).pow(2))

    # Compute the Gaussian mask
    mask = torch.exp(-0.5 * d_square)

    # Adjust shape for multiple channels if necessary
    if len(shape) > 2:
        C = shape[1]
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    # print(mask.shape)
    return mask


def freq_mix_2d(x, noise, LPF):
    """
    Noise reinitialization.

    Args:
        x: diffused latent
        noise: randomly sampled noise
        LPF: low pass filter
    """
    # FFT
    x_freq = fft.fft2(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fft2(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))

    LPF = LPF.to(x_freq.device)
    # frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high # mix in freq domain

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real

    return x_mixed

def sample_in_neighborhood_normal(noise, d):
    """
    在噪声的d邻域内采样，保证结果仍然服从标准正态分布N(0,1)
    
    Args:
        noise: 原始噪声张量 [C, H, W]，服从N(0,1)
        d: 邻域半径（0到1之间）
    
    Returns:
        新的噪声张量，服从N(0,1)且在原始噪声的d邻域内
    """
    # 生成新的标准正态分布噪声
    new_noise = torch.randn_like(noise)
    
    # 使用插值来保证结果在d邻域内且服从标准正态分布
    # alpha的大小决定了新采样点与原始点的距离
    alpha = torch.rand_like(noise)

    # alpha = alpha.to(noise.device)
    
    # 使用球面插值确保结果仍然服从标准正态分布
    interpolated_noise = (1 - alpha) * noise + alpha * new_noise
    
    # 重新归一化以确保方差为1
    norm_factor = torch.sqrt((1 - alpha)**2 + alpha**2)
    normalized_noise = interpolated_noise / norm_factor
    
    return normalized_noise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = torch.arange(50).view(1, 2, 1, 5, 5).to(device)
# x = x.float()
# # noise = torch.randn_like(x)

# x = sample_in_neighborhood_normal(x, 0.1)
# print(x.shape)
# LPF = gaussian_low_pass_filter_2d(x.shape, 0.25)
# x_mixed = freq_mix_2d(x, noise, LPF)

# print(x_mixed.shape)
