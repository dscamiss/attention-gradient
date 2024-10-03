"""Scaled dot-product attention map."""

import math

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class Attention(Function):
    """Scaled dot-product attention map."""

    @staticmethod
    def forward(ctx: FunctionCtx, theta: Float[Tensor, "n 3d"]) -> Float[Tensor, "n d"]:
        """Compute scaled dot-product attention map output.

        Args:
            ctx (FunctionCtx): Context used to stash data for `backward()`.
            theta (Tensor): Input tensor of shape `(n, 3d)`; `theta = [Q, K, V]`.
        """
        d = theta.shape[-1] // 3
        q, k, v = theta.split(d, dim=-1)
        s = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)
        ctx.save_for_backward(q, k, v, s)
        return s @ v

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: FunctionCtx, grad_output: Float[Tensor, "n d"]
    ) -> Float[Tensor, "n 3d"]:
        """Compute gradient of scaled dot-product attention map.

        Args:
            ctx (FunctionCtx): Context used to retrieve stashed data from `forward()`.
            grad_output (Tensor): Gradient tensor of shape `(n, 3d)`.
        """
        q, k, v, s = ctx.saved_tensors  # type: ignore[attr-defined]
        n, d = q.shape
        v_transpose = v.transpose(-1, -2)

        # Compute "omega" matrix, row-by-row
        omega = torch.zeros(n, n, dtype=torch.double)
        for i in range(n):
            s_row = s[i, :].unsqueeze(0)
            s_col = s_row.transpose(-1, -2)
            dsigma = torch.diag(s_col.squeeze()) - (s_col @ s_row)
            omega[i, :] = grad_output[i, :].unsqueeze(0) @ v_transpose @ dsigma

        # Incorporate scaling factor
        omega = (1.0 / math.sqrt(d)) * omega

        # Compute "Q" component
        q_comp = omega @ k

        # Compute "K" component
        k_comp = omega.transpose(-1, -2) @ q

        # Compute "V" component
        v_comp = torch.zeros(n, d, dtype=torch.double)
        for i in range(n):
            s_col = s[i, :].unsqueeze(0).transpose(-1, -2)
            v_comp += s_col @ grad_output[i, :].unsqueeze(0)

        # Gradient concatenates all components
        return torch.cat((q_comp, k_comp, v_comp), dim=-1)
