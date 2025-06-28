# Copyright © 2024 Apple Inc.

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import os
import mlx.core as mx
import mlx.nn as nn


from .chromalayers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    SingleStreamBlock,
    Approximator,
    ChromaModulationOut,
    timestep_embedding,
)
def debug_print(x, name="array"):
    import numpy as np
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    print(f"{name}: shape = {x.shape()}")
    print(x.numpy())

@dataclass
class ChromaParams:
    in_channels: int
    out_channels: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    patch_size: int
    qkv_bias: bool
    in_dim: int
    out_dim: int
    hidden_dim: int
    n_layers: int



class Chroma(nn.Module):
    def __init__(self, params: ChromaParams):
        super().__init__()

        self.params = params
        self.patch_size = params.patch_size
        self.context_in_dim = params.context_in_dim
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads

        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.in_dim = params.in_dim
        self.out_dim = params.out_dim
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.distilled_guidance_layer = Approximator(
                    in_dim=self.in_dim,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                    n_layers=self.n_layers,
                )

        self.txt_in = nn.Linear(self.context_in_dim, self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=params.mlp_ratio,
                qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
            )
            for _ in range(params.depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.dtype = mx.bfloat16

    def sanitize(self, weights):
        new_weights = {}
        for k, w in weights.items():
            before_key = k
            if k.endswith(".scale") :
                k = k[:-6] + ".weight"
            for seq in ["img_mlp", "txt_mlp"]:
                if f".{seq}." in k:
                    k = k.replace(f".{seq}.", f".{seq}.layers.")
                    break
            if ".encoder.blocks." in k:
                k = k.replace(".encoder.blocks.", ".encoder.layers.")
            if ".layers.layers." in k:
                k = k.replace(".layers.layers.", ".layers.")
            new_weights[k] = w
            after_key = k
        return new_weights
    
    def get_modulations(self, tensor: mx.array, block_type: str, *, idx: int = 0):
        # This function slices up the modulations tensor which has the following layout:
        #   single     : num_single_blocks * 3 elements
        #   double_img : num_double_blocks * 6 elements
        #   double_txt : num_double_blocks * 6 elements
        #   final      : 2 elements
        # print("tensor :", tensor)
        if block_type == "final":
            return (tensor[:, -2:-1, :], tensor[:, -1:, :])
        single_block_count = self.params.depth_single_blocks
        double_block_count = self.params.depth
        offset = 3 * idx
        if block_type == "single":
            return ChromaModulationOut.from_offset(tensor, offset)
        # Double block modulations are 6 elements so we double 3 * idx.
        offset *= 2
        if block_type in {"double_img", "double_txt"}:
            # Advance past the single block modulations.
            offset += 3 * single_block_count
            if block_type == "double_txt":
                # Advance past the double block img modulations.
                offset += 6 * double_block_count
            # print("offset", offset)
            return (
                ChromaModulationOut.from_offset(tensor, offset),
                ChromaModulationOut.from_offset(tensor, offset + 3),
            )
        raise ValueError("Bad block_type")
    
    def __call__(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt: mx.array,
        txt_ids: mx.array,
        timesteps: mx.array,
        guidance: Optional[mx.array] = None,
    ) -> mx.array:
        
        if img.ndim != 3 or txt.ndim != 3:
            print(img.ndim, txt.ndim)
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        batch_size = img.shape[0]
        img = self.img_in(img)
        self.dtype = img.dtype
        
        # 产生调制向量过程
        mod_index_length = 344
        distill_timestep = timestep_embedding(mx.stop_gradient(timesteps), 16).astype(self.dtype)
        distil_guidance = timestep_embedding(mx.stop_gradient(guidance), 16).astype(self.dtype)

        
        # get all modulation index
        
        modulation_index = timestep_embedding(mx.arange(mod_index_length, dtype=mx.float32), 32).astype(self.dtype) #这里非常重要，mx.arange默认是 int会导致后面的除第一个以外，后面的向量都相同。所以要置顶数据类型.bf16会导致乱码.float32可以出现虚影。
        modulation_index = mx.broadcast_to(modulation_index, (batch_size, mod_index_length, 32)).astype(self.dtype)  # (B, 344, 32)
        # mx.save("mlx_modulation_index",modulation_index.astype(mx.float32))
        ## 已验证，与ComfyUI生成版本接近

        timestep_guidance = mx.concatenate([distill_timestep, distil_guidance], axis=1).astype(self.dtype)
        timestep_guidance = timestep_guidance[:, None, :]  # shape: (B, 1, 32)
        timestep_guidance = mx.broadcast_to(timestep_guidance, (batch_size, mod_index_length, 32)).astype(self.dtype)  # (B, 344, 32)
        # mx.save("mlx_timestep_guidance",timestep_guidance.astype(mx.float32))
        
        
        # Match img's dtype
        # timestep_guidance = timestep_guidance.astype(img.dtype)
        # then and only then we could concatenate it together
        
        
        input_vec = mx.concatenate([timestep_guidance, modulation_index], axis=-1).astype(self.dtype)
        # mx.save("mlx_input_vec",input_vec.astype(mx.float32))
        
        
        mod_vectors = self.distilled_guidance_layer(input_vec).astype(self.dtype)
        
        # mx.save("mlx_mod_vectors",mod_vectors.astype(mx.float32))
        
        txt = self.txt_in(txt).astype(self.dtype)
        # print("txt_in shape:",txt.shape)
        # mx.save("mlx_txt_in",txt.astype(mx.float32))
        ids = mx.concatenate([txt_ids, img_ids], axis=1)
        pe = self.pe_embedder(ids).astype(img.dtype)
        # mx.save("mlx_pe",pe.astype(mx.float32))
        
        for i,block in enumerate(self.double_blocks):
            # print(f"Double block {i} linear1 weight min/max:", mx.min(block.img_norm1.weight), mx.max(block.img_norm1.weight))
            double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
            # (img_mod1, img_mod2), (txt_mod1, txt_mod2) = double_mod
            # mx.save(f"mlx_double_block_{i}_img_mod1_shift",img_mod1.shift.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_img_mod1_scale",img_mod1.scale.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_img_mod2_shift",img_mod2.shift.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_img_mod2_scale",img_mod2.scale.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_txt_mod1_shift",txt_mod1.shift.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_txt_mod1_scale",txt_mod1.scale.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_txt_mod2_shift",txt_mod2.shift.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_txt_mod2_scale",txt_mod2.scale.astype(mx.float32))
            # print(f"Block {i}")
            img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe)
            # mx.save(f"mlx_double_block_{i}_img", img.astype(mx.float32))
            # mx.save(f"mlx_double_block_{i}_txt", txt.astype(mx.float32))
            
        img = mx.concatenate([txt, img], axis=1)
        # mx.save("mlx_before_single_img", img.astype(mx.float32))

        # print("After Double Blockimg 前 24 个元素:", img.flatten()[:24],mx.min(img), mx.max(img))
        for i, block in enumerate(self.single_blocks):
            
            single_mod = self.get_modulations(mod_vectors, "single", idx=i)
            # mx.save(f"mlx_single_block_{i}_mod_shift", single_mod.shift.astype(mx.float32))
            # mx.save(f"mlx_single_block_{i}_mod_scale", single_mod.scale.astype(mx.float32))
            
            img = block(img, vec=single_mod, pe=pe)
            # mx.save(f"mlx_single_block_{i}_img", img.astype(mx.float32))
            
        
        
        img = img[:, txt.shape[1] :, ...]
        # mx.save("mlx_before_final_img", img.astype(mx.float32))

        final_mod = self.get_modulations(mod_vectors, "final")
        # shift, scale = final_mod
        
        # shift = mx.squeeze(shift, axis=1)
        # scale = mx.squeeze(scale, axis=1)
        # mx.save("mlx_final_layer_shift", shift.astype(mx.float32))
        # mx.save("mlx_final_layer_scale", scale.astype(mx.float32))
        img = self.final_layer(img, final_mod)
        # save_dir = "mlx_final"
        # name = "mlx_after_final_img"
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_path = os.path.join(save_dir, f"{name}_{timestamp}.npy")
        # mx.save(save_path, img.astype(mx.float32))
        return img
