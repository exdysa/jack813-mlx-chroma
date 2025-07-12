# Copyright © 2024 Apple Inc.

from typing import Tuple
import numpy as np

from chroma.chromasampler import ChromaSampler
from chroma.lora import LoRALinear
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from tqdm import tqdm

from .utils import (
    load_ae,
    load_chroma_model,
    load_t5,
    load_t5_tokenizer,
)

def load_debug_data(name):
    np_arr = np.load(f"debugs/{name}.npy")
    # 2. 转成 mlx.core.array
    mx_arr = mx.array(np_arr)

    # 3. 转成 bfloat16（MLX 支持的 bf16 类型）
    mx_arr_bf16 = mx_arr.astype(mx.bfloat16)
    return mx_arr_bf16

class ChromaPipeline:
    def __init__(self, name: str, download_hf: bool = True, chroma_filepath: str = None, t5_filepath:str = None, tokenizer_filepath:str = None, vae_filepath:str = None, load_quantized=False):
        chroma_fp = chroma_filepath
        t5_fp = t5_filepath
        tokenizer_fp = tokenizer_filepath
        vae_fp = vae_filepath
        if download_hf:
            chroma_fp = None
            t5_fp = None
            tokenizer_fp = None
            vae_fp = None
        self.dtype = mx.bfloat16
        self.name = name
        self.t5_padding = False
        self.flow = load_chroma_model(name, chroma_fp, quantized=load_quantized)
        self.t5 = load_t5(name,file_path=t5_fp,quantized=load_quantized)
        self.t5_tokenizer = load_t5_tokenizer(name, file_path=tokenizer_fp)
        self.ae = load_ae(name, file_path = vae_fp)
        self.sampler = ChromaSampler(name)
        

    def ensure_models_are_loaded(self):
        mx.eval(
            self.ae.parameters(),
            self.flow.parameters(),
            self.t5.parameters(),
        )

    def reload_text_encoders(self):
        self.t5 = load_t5(self.name)

    def tokenize(self, text):
        t5_tokens = self.t5_tokenizer.encode(text, pad=self.t5_padding)
        
        return t5_tokens

    def _prepare_latent_images(self, x):
        b, h, w, c = x.shape

        # Pack the latent image to 2x2 patches
        x = x.reshape(b, h // 2, 2, w // 2, 2, c)
        x = x.transpose(0, 1, 3, 5, 2, 4).reshape(b, h * w // 4, c * 4)

        # Create positions ids used to positionally encode each patch. Due to
        # the way RoPE works, this results in an interesting positional
        # encoding where parts of the feature are holding different positional
        # information. Namely, the first part holds information independent of
        # the spatial position (hence 0s), the 2nd part holds vertical spatial
        # information and the last one horizontal.
        i = mx.zeros((h // 2, w // 2), dtype=mx.int32)
        j, k = mx.meshgrid(mx.arange(h // 2), mx.arange(w // 2), indexing="ij")
        x_ids = mx.stack([i, j, k], axis=-1)
        x_ids = mx.repeat(x_ids.reshape(1, h * w // 4, 3), b, 0)

        return x, x_ids

    def _prepare_conditioning(self, n_images, t5_tokens ):
        # Prepare the text features
        txt = self.t5(t5_tokens)
        if len(txt) == 1 and n_images > 1:
            txt = mx.broadcast_to(txt, (n_images, *txt.shape[1:]))
        txt_ids = mx.zeros((n_images, txt.shape[1], 3), dtype=mx.int32)

        return txt, txt_ids

    def _denoising_loop(
        self,
        x_t,
        x_ids,
        txt,
        txt_ids,
        neg_txt,
        neg_txt_ids,
        num_steps: int = 28,
        guidance: float = 0.0,
        start: float = 1,
        stop: float = 0,
        first_n_steps_without_cfg : int = 0,
        cfg: float = 4.0
    ):
        B = len(x_t)

        def scalar(x):
            return mx.full((B,), x, dtype=self.dtype)
        
        guidance = scalar(guidance)
        timesteps = self.sampler.timesteps(
            num_steps,
            x_t.shape[1],
            start=start,
            stop=stop,
        )
       
        for i in range(num_steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            pred = self.flow(
                img=x_t,
                img_ids=x_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=scalar(t),
                guidance=guidance,
            )
            if i < first_n_steps_without_cfg or first_n_steps_without_cfg == -1:
                x_t = x_t.astype(pred.dtype) + (t_prev - t) * pred
            else:
                pred_neg = self.flow(
                    img=x_t,
                    img_ids=x_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    timesteps=scalar(t),
                    guidance=guidance,
                )

                pred_cfg = pred_neg + (pred - pred_neg) * cfg

                x_t = self.sampler.step(pred_cfg, x_t, t, t_prev)

            yield x_t

    def generate_latents(
        self,
        text: str,
        neg_text:str,
        n_images: int = 1,
        num_steps: int = 28,
        guidance: float = 0.0,
        latent_size: Tuple[int, int] = (64, 64),
        seed = None,
        first_n_steps_without_cfg = 4,
        cfg = 2.0
    ):
        # Set the PRNG state
        if seed is not None:
            mx.random.seed(seed)

        # Create the latent variables
        img_vec = self.sampler.sample_prior((n_images, *latent_size, 16), dtype=mx.bfloat16)
        img_vec, img_ids = self._prepare_latent_images(img_vec)
        
        print("img Vec Shape",img_vec.shape)
        # Get the conditioning
        t5_tokens = self.tokenize(text)
        neg_t5_tokens = self.tokenize(neg_text)
        txt, txt_ids = self._prepare_conditioning(n_images, t5_tokens)
        neg_txt,neg_txt_ids = self._prepare_conditioning(n_images, neg_t5_tokens)
        # Yield the conditioning for controlled evaluation by the caller
        
        yield (img_vec, img_ids, txt, txt_ids, neg_text, neg_txt_ids)
        # Yield the latent sequences from the denoising loop

        yield from self._denoising_loop(
            img_vec, img_ids, txt, txt_ids , neg_txt, neg_txt_ids,  num_steps=num_steps, guidance=guidance , first_n_steps_without_cfg = first_n_steps_without_cfg, cfg = cfg
        )

    def decode(self, x, latent_size: Tuple[int, int] = (64, 64)):
        h, w = latent_size
        
        x = x.reshape(len(x), h // 2, w // 2, -1, 2, 2)
        x = x.transpose(0, 1, 4, 2, 5, 3).reshape(len(x), h, w, -1)
        x = self.ae.decode(x)
        x = mx.clip(x + 1, 0, 2) * 0.5
        return x
    


    
    def linear_to_lora_layers(self, rank: int = 8, num_blocks: int = -1):
        """Swap the linear layers in the transformer blocks with LoRA layers."""
        all_blocks = self.flow.double_blocks + self.flow.single_blocks
        all_blocks.reverse()
        num_blocks = num_blocks if num_blocks > 0 else len(all_blocks)
        
        for i, block in zip(range(num_blocks), all_blocks):
            # replace_linear_recursive_mlx(block, rank)
            loras = []
            for name, module in block.named_modules():
               
                if isinstance(module, nn.Linear):
                    
                    # print(f"add {name} LoRALinear")
                    loras.append((name, LoRALinear.from_base(module, r=rank)))
                 
            block.update_modules(tree_unflatten(loras))
            
            

    def fuse_lora_layers(self):
        fused_layers = []
        for name, module in self.flow.named_modules():
            if isinstance(module, LoRALinear):
                print(f"[CHECK] Found LoRA layer at: {name}")
                print(f"  - lora_a Mean(abs): {mx.mean(mx.abs(module.lora_a)).item():.6f}")
                print(f"  - lora_b Mean(abs): {mx.mean(mx.abs(module.lora_b)).item():.6f}")
                fused_layers.append((name, module.fuse()))
        self.flow.update_modules(tree_unflatten(fused_layers))

    # def generate_images(
    #     self,
    #     text: str,
    #     n_images: int = 1,
    #     num_steps: int = 35,
    #     guidance: float = 4.0,
    #     latent_size: Tuple[int, int] = (64, 64),
    #     seed=None,
    #     reload_text_encoders: bool = True,
    #     progress: bool = True,
    # ):
    #     latents = self.generate_latents(
    #         text, n_images, num_steps, guidance, latent_size, seed
    #     )
    #     mx.eval(next(latents))

    #     if reload_text_encoders:
    #         self.reload_text_encoders()

    #     for img_vec in tqdm(latents, total=num_steps, disable=not progress, leave=True):
    #         mx.eval(img_vec)

    #     images = []
    #     for i in tqdm(range(len(img_vec)), disable=not progress, desc="generate images"):
    #         images.append(self.decode(img_vec[i : i + 1]))
    #         mx.eval(images[-1])
    #     images = mx.concatenate(images, axis=0)
        
    #     mx.eval(images)
    #     return images

def set_module_by_path(root, path, new_module):
    parts = path.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            root = root[int(p)]
        else:
            root = getattr(root, p)
    last = parts[-1]
    if last.isdigit():
        root[int(last)] = new_module
    else:
        setattr(root, last, new_module)