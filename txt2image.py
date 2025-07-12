# Copyright © 2024 Apple Inc.

import argparse
import time
from chroma.lora import LoRALinear
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from chroma import ChromaPipeline


def to_latent_size(image_size):
    w, h = image_size
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16

    if (h, w) != image_size:
        print(
            "Warning: The image dimensions need to be divisible by 16px. "
            f"Changing size to {w}x{h}."
        )

    return (h // 8, w // 8)


def quantization_predicate(name, m):
    return hasattr(m, "to_quantized") and m.weight.shape[1] % 512 == 0



def load_adapter(chroma, adapter_file, fuse=False):
    weights, lora_config = mx.load(adapter_file, return_metadata=True)
    print('lora_config', lora_config)
    # rank = int(lora_config["lora_rank"])
    # rank = int(lora_config.get("ss_network_dim", 16))
        # 自动推断
    num_blocks = -1
    a = weights["diffusion_model.double_blocks.0.img_mlp.2.lora_A.weight"]
    b = weights["diffusion_model.double_blocks.0.img_mlp.2.lora_B.weight"]
    rank = a.shape[0] if a.shape[0] == b.shape[1] else a.shape[1]
    # print("Detected LoRA rank:", r)
    print(f"[INFO] Using LoRA rank: {rank}")
    chroma.linear_to_lora_layers(rank, num_blocks)
    # 调试类型

    new_weights = {}
    for k, v in weights.items():
        # 删除前缀 diffusion_model.
        if k.startswith("diffusion_model."):
            new_k = k[len("diffusion_model."):]
        else:
            new_k = k
        # 替换 .lora_A.weight -> .lora_a
        new_k = new_k.replace(".txt_mlp", ".txt_mlp.layers")
        new_k = new_k.replace(".img_mlp", ".img_mlp.layers")
        new_k = new_k.replace(".lora_A.weight", ".lora_a")
        new_k = new_k.replace(".lora_B.weight", ".lora_b")
        new_weights[new_k] = v
    chroma.flow.load_weights(list(new_weights.items()), strict=False)
    if fuse:
        chroma.fuse_lora_layers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate images from a textual prompt using stable diffusion"
    )
    parser.add_argument("prompt")
    parser.add_argument("--neg-prompt", default="")
    parser.add_argument("--download-hf", type=bool, default=False)
    parser.add_argument("--chroma-path", default="./models/chroma/chroma.safetensors")
    parser.add_argument("--t5-path", default="./models/t5/text_encoder_2")
    parser.add_argument("--tokenizer-path", default="./models/t5/tokenizer_2")
    parser.add_argument("--vae-path", default="./models/vae")
    parser.add_argument("--n-images", type=int, default=1)
    parser.add_argument(
        "--image-size", type=lambda x: tuple(map(int, x.split("x"))), default=(512, 512)
    )
    parser.add_argument("--steps", type=int,default=28)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--skip-cfg-steps", type=int, default=0)
    parser.add_argument("--n-rows", type=int, default=1)
    parser.add_argument("--decoding-batch-size", type=int, default=1)
    parser.add_argument("--quantize", "-q", type=bool, default=False)
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--seed", type=int, default=666)

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--adapter")
    parser.add_argument("--fuse-adapter", action="store_true")
    # parser.add_argument("--no-t5-padding", dest="t5_padding", action="store_false")
    args = parser.parse_args()
    # print(args.n_images)
    # Load the models

    chroma = ChromaPipeline("chroma", download_hf=args.download_hf, chroma_filepath=args.chroma_path, t5_filepath=args.t5_path, tokenizer_filepath=args.tokenizer_path, vae_filepath=args.vae_path, load_quantized=args.quantize)
    if args.adapter:
        load_adapter(chroma, args.adapter, fuse=args.fuse_adapter)
    if args.preload_models:
        chroma.ensure_models_are_loaded()

    # Make the generator
    latent_size = to_latent_size(args.image_size)
    latents = chroma.generate_latents(
        args.prompt,
        args.neg_prompt,
        n_images=args.n_images,
        num_steps=args.steps,
        latent_size=latent_size,
        seed=args.seed,
        first_n_steps_without_cfg = args.skip_cfg_steps,
        cfg=args.cfg,

    )

    # First we get and eval the conditioning
    conditioning = next(latents)
    mx.eval(conditioning)
    peak_mem_conditioning = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # The following is not necessary but it may help in memory constrained
    # systems by reusing the memory kept by the text encoders.
    del chroma.t5
    
    start = time.perf_counter()
    # Actual denoising loop
    for x_t in tqdm(latents, total=args.steps):
        mx.eval(x_t)

    # The following is not necessary but it may help in memory constrained
    # systems by reusing the memory kept by the flow transformer.
    del chroma.flow
    peak_mem_generation = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, args.n_images, args.decoding_batch_size)):
        decoded.append(chroma.decode(x_t[i : i + args.decoding_batch_size], latent_size))
        mx.eval(decoded[-1])
    peak_mem_decoding = mx.get_peak_memory() / 1024**3
    peak_mem_overall = max(
        peak_mem_conditioning, peak_mem_generation, peak_mem_decoding
    )

    if args.save_raw:
        *name, suffix = args.output.split(".")
        name = ".".join(name)
        x = mx.concatenate(decoded, axis=0)
        x = (x * 255).astype(mx.uint8)
        for i in range(len(x)):
            im = Image.fromarray(np.array(x[i]))
            im.save(".".join([name, str(i), suffix]))
    else:
        # Arrange them on a grid
        x = mx.concatenate(decoded, axis=0)
        x = mx.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)])
        B, H, W, C = x.shape
        x = x.reshape(args.n_rows, B // args.n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(args.n_rows * H, B // args.n_rows * W, C)
        x = (x * 255).astype(mx.uint8)

        # Save them to disc
        im = Image.fromarray(np.array(x))
        im.save(args.output)
    end = time.perf_counter()
    elapsed_time = end - start
    # Report the peak memory used during generation
    if args.verbose:
        print(f"Peak memory used for the text:       {peak_mem_conditioning:.3f} GB")
        print(f"Peak memory used for the generation: {peak_mem_generation:.3f} GB")
        print(f"Peak memory used for the decoding:   {peak_mem_decoding:.3f} GB")
        print(f"Peak memory used overall:            {peak_mem_overall:.3f} GB")
        print(f"Prompt execution time:               {elapsed_time:.4f} Seconds")
