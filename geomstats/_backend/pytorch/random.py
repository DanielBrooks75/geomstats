"""Torch based random backend."""

import torch


def multivariate_normal(*args, **kwargs):
    return torch.distributions.multivariate_normal.MultivariateNormal(
        *args, **kwargs)


def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, '__iter__'):
        size = (size,)
    return torch.normal(mean=loc, std=scale, size=size)


permutation = torch.randperm


def rand(*args, **kwargs):
    return torch.rand(*args, **kwargs)


def randn(*args, **kwargs):
    return torch.randn(*args, **kwargs)


def randint(*args, **kwargs):
    return torch.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return torch.manual_seed(*args, **kwargs)


def shuffle(x):
    n = x.shape[0]
    idx = torch.randperm(n)
    return x[idx]


def uniform(low=0.0, high=1.0, size=(1,)):
    if not hasattr(size, '__iter__'):
        size = (size,)
    if low >= high:
        raise ValueError('Upper bound must be higher than lower bound')
    return (high - low) * torch.rand(*size) + low
