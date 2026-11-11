"""Distinct CUDA streams for independent reparameterization noise per replica."""
import torch


def seed_gaussian_streams(seed, mode):
    if mode == 'legacy':
        return  # setup_seed already assigned the same seed to every CUDA device.
    if mode != 'independent':
        raise ValueError('Unknown Gaussian stream mode: ' + mode)
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.manual_seed((seed + 1000003 * index) % (2 ** 63 - 1))
