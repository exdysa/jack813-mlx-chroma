# Copyright © 2024 Apple Inc.

import math

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    @staticmethod
    def from_base(
        linear: nn.Linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 2.0,
    ):
        # print(f"[DEBUG] from_base called: {linear=}, r={r}")
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear

        # ✅ 添加 shape 检查并修复 lora_a
        if lora_lin.lora_a.shape != (input_dims, r):
            if lora_lin.lora_a.shape == (r, input_dims):
                print(f"[Fixing Shape] Transposing lora_a from {lora_lin.lora_a.shape}")
                lora_lin.lora_a = lora_lin.lora_a.T
            else:
                raise ValueError(
                    f"[LoRA Error] Unexpected lora_a shape: {lora_lin.lora_a.shape}, expected ({input_dims}, {r})"
                )

        return lora_lin

    def fuse(self):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        dtype = weight.dtype

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = self.scale * self.lora_b.T
        lora_a = self.lora_a.T
        fused_linear.weight = weight + (lora_b.T @ lora_a.T).astype(dtype)
        if bias:
            fused_linear.bias = linear.bias

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

        # Scale for low-rank update
        self.scale = scale

        # Low rank lora weights
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a.T) @ self.lora_b.T
        return y + (self.scale * z).astype(x.dtype)