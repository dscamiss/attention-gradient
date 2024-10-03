"""Test code for `attention.py`."""

import torch
from torch.autograd import gradcheck

from attention import Attention


def test_attention_gradient() -> None:
    """Check correctness of `Attention.backward()` using `gradcheck()`."""
    n, d = 8, 16

    q = torch.randn(n, d, dtype=torch.double, requires_grad=True)
    k = torch.randn(n, d, dtype=torch.double, requires_grad=True)
    v = torch.randn(n, d, dtype=torch.double, requires_grad=True)

    theta = torch.cat((q, k, v), dim=-1)

    if gradcheck(Attention.apply, theta, eps=1e-6, atol=1e-4):
        print("success!")
    else:
        print("failure...")
