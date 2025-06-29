# Copyright © 2024 Apple Inc.

import json
import os
from dataclasses import dataclass
from typing import Optional

from chroma.chromamodel import Chroma, ChromaParams
import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from .autoencoder import AutoEncoder, AutoEncoderParams
from .t5 import T5Config, T5Encoder
from .tokenizers import CLIPTokenizer, T5Tokenizer


@dataclass
class ModelSpec:
    params: ChromaParams
    ae_params: AutoEncoderParams
    ckpt_path: Optional[str]
    ae_path: Optional[str]
    repo_id: Optional[str]
    repo_flow: Optional[str]
    repo_ae: Optional[str]


configs = {
    "chroma": ModelSpec(
        repo_id="jack813liu/mlx-chroma",
        repo_flow="chroma/chroma.safetensors",
        repo_ae="vae/ae.safetensors",
        ckpt_path=os.getenv("chroma"),
        params=ChromaParams(
            in_channels= 64,
            out_channels = 64,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            patch_size = 2,
            qkv_bias=True,
            in_dim = 64,
            out_dim = 3072,
            hidden_dim = 5120,
            n_layers = 5
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

def strip_prefix(state_dict, prefix_to_remove):
    """移除 state_dict 中所有 key 的指定前缀"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_chroma_model(name: str, hf_download: bool = True, file_path: str=None, quantized=False):
    model = Chroma(configs["chroma"].params)
    
    print(file_path)
    # Load the checkpoint if needed
    if file_path is not None:
        ckpt_path = file_path
    else:
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)
    raw_weights = mx.load(ckpt_path)
        # 去除前缀（你可以修改为你实际需要移除的）
    clean_weights = strip_prefix(raw_weights, "model.diffusion_model.")
    weights = model.sanitize(clean_weights)
    model.load_weights(list(weights.items()))
    if  quantized:
        nn.quantize(model, bits=8, group_size=64)
    return model


def load_ae(name: str, hf_download: bool = True, file_path: str = None):
    # Get the safetensors file to load
    if file_path is not None:
        ckpt_path = f"{file_path}/ae.safetensors"
    else:
        ckpt_path = configs[name].ae_path
        # Download if needed
        if (
            ckpt_path is None
            and configs[name].repo_id is not None
            and configs[name].repo_ae is not None
            and hf_download
        ):
            ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Make the autoencoder
    ae = AutoEncoder(configs[name].ae_params)
    # Load the checkpoint if needed
    if ckpt_path is not None:
        weights = mx.load(ckpt_path)
        weights = ae.sanitize(weights)
        ae.load_weights(list(weights.items()))

    return ae



def load_t5(name: str= None, file_path:str = None, quantized: bool = False):
    # Load the config
    t5_config = {
        "_name_or_path": "google/t5-v1_1-xxl",
        "architectures": [
            "T5EncoderModel"
        ],
        "classifier_dropout": 0.0,
        "d_ff": 10240,
        "d_kv": 64,
        "d_model": 4096,
        "decoder_start_token_id": 0,
        "dense_act_fn": "gelu_new",
        "dropout_rate": 0.1,
        "eos_token_id": 1,
        "feed_forward_proj": "gated-gelu",
        "initializer_factor": 1.0,
        "is_encoder_decoder": True,
        "is_gated_act": True,
        "layer_norm_epsilon": 1e-06,
        "model_type": "t5",
        "num_decoder_layers": 24,
        "num_heads": 64,
        "num_layers": 24,
        "output_past": True,
        "pad_token_id": 0,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.3",
        "use_cache": True,
        "vocab_size": 32128
        }

    # Make the T5 model
    config = T5Config.from_dict(t5_config)
    t5 = T5Encoder(config)
    
    # Load the weights
    if file_path is not None:
        model_index = file_path + "/model.safetensors.index.json"
        print("T5 model index:",model_index)
        weight_files = set()
        with open(model_index) as f:
            for _, w in json.load(f)["weight_map"].items():
                weight_files.add(w)
        weights = {}
        for w in weight_files:
            w = f"{file_path}/{w}"

            weights.update(mx.load(w))
    else:
        model_index = hf_hub_download(
                configs[name].repo_id, "t5/text_encoder_2/model.safetensors.index.json"
            )
        print("T5 model index:",model_index)
        weight_files = set()
        with open(model_index) as f:
            for _, w in json.load(f)["weight_map"].items():
                weight_files.add(w)
        weights = {}
        for w in weight_files:
            w = f"t5/text_encoder_2/{w}"
            w = hf_hub_download(configs[name].repo_id, w)
            weights.update(mx.load(w))
    weights = t5.sanitize(weights)
    t5.load_weights(list(weights.items()))
    if quantized:
        nn.quantize(t5, bits=8, group_size=64)
    return t5



def load_t5_tokenizer(name: str, pad: bool = True,file_path: str = None):
    if file_path is not None:
        model_file = f"{file_path}/spiece.model"
    else:
        model_file = hf_hub_download(configs[name].repo_id, "t5/tokenizer_2/spiece.model")
    return T5Tokenizer(model_file, 512)
