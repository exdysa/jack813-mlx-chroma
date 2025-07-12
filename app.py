from chroma.chroma import ChromaPipeline
import gradio as gr
import mlx.core as mx
import time
from PIL import Image
from tqdm import tqdm
import numpy as np
from txt2image import load_adapter, to_latent_size


def generate(prompt, 
             neg_prompt,
             width=512,
             height=512,
             seed=666, 
             steps=28,
             first_n_steps_without_cfg=0,
             cfg=4.0,
             n_images=1, 
             n_rows=1, 
             decoding_batch_size=1, 
             output="out.png", 
             save_raw=False, 
             download_hf=False, 
             chroma_path="./models/chroma/chroma.safetensors",
             t5_path="./models/t5/text_encoder_2",
             tokenizer_path="./models/t5/tokenizer_2",
             vae_path="./models/vae",
             quantize=False,
             lora_path=None,
             random_seed_check=False,
             verbose=False):
    chroma = ChromaPipeline("chroma", download_hf=download_hf, chroma_filepath=chroma_path, t5_filepath=t5_path, tokenizer_filepath=tokenizer_path, vae_filepath=vae_path, load_quantized=quantize)
    if lora_path:
        load_adapter(chroma, lora_path, fuse=False)
    # if args.preload_models:
    #     chroma.ensure_models_are_loaded()
    if random_seed_check:
        seed = str(np.random.randint(0, 1000000))
    # Make the generator
    latent_size = to_latent_size((int(width),int(height)))
    latents = chroma.generate_latents(
        prompt,
        neg_prompt,
        n_images=n_images,
        num_steps=steps,
        latent_size=latent_size,
        seed=seed,
        first_n_steps_without_cfg = first_n_steps_without_cfg,
        cfg=cfg,

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
    for x_t in tqdm(latents, total=steps):
        mx.eval(x_t)

    # The following is not necessary but it may help in memory constrained
    # systems by reusing the memory kept by the flow transformer.
    del chroma.flow
    peak_mem_generation = mx.get_peak_memory() / 1024**3
    mx.reset_peak_memory()

    # Decode them into images
    decoded = []
    for i in tqdm(range(0, n_images, decoding_batch_size)):
        decoded.append(chroma.decode(x_t[i : i + decoding_batch_size], latent_size))
        mx.eval(decoded[-1])
    peak_mem_decoding = mx.get_peak_memory() / 1024**3
    peak_mem_overall = max(
        peak_mem_conditioning, peak_mem_generation, peak_mem_decoding
    )

    if save_raw:
        *name, suffix = output.split(".")
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
        x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(n_rows * H, B // n_rows * W, C)
        x = (x * 255).astype(mx.uint8)

        # Save them to disc
        im = Image.fromarray(np.array(x))
        im.save(output)
    end = time.perf_counter()
    elapsed_time = end - start
    # Report the peak memory used during generation
    if verbose:
        print(f"Peak memory used for the text:       {peak_mem_conditioning:.3f} GB")
        print(f"Peak memory used for the generation: {peak_mem_generation:.3f} GB")
        print(f"Peak memory used for the decoding:   {peak_mem_decoding:.3f} GB")
        print(f"Peak memory used overall:            {peak_mem_overall:.3f} GB")
        print(f"Prompt execution time:               {elapsed_time:.4f} Seconds")
    return im , seed
    # return "Hello " + prompt + neg_prompt + "!"

with gr.Blocks() as demo:
    gr.Markdown("## MLX-Chroma")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Text to Image Generation")
            prompt = gr.Textbox("", label="Enter your prompt here", placeholder="Type something...")
            neg_prompt = gr.Textbox("", label="Negtive Prompt", placeholder="Generated output will appear here")
            with gr.Row():
                width = gr.Textbox("512", label="Width", placeholder="Width of the generated image")
                height = gr.Textbox("512", label="Height", placeholder="Height of the generated image")
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    seed = gr.Textbox("666", label="Seed", placeholder="Random seed for generation")
                    random_seed_check = gr.Checkbox(True, label="Random Seed", info="Generate a random seed for each generation")
                with gr.Column(scale=1, min_width=100):
                    steps = gr.Slider(1, 100, value=28, label="Steps", info="Number of denoising steps")
            with gr.Row():
                first_n_steps_without_cfg = gr.Slider(0, 20, value=0, label="First N Steps without CFG", info="Number of steps to run without classifier-free guidance")
                cfg = gr.Slider(0.0, 20.0, value=4.0, label="CFG Scale", info="Classifier-Free Guidance scale")
            
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            lora_path = gr.File(None, label="LoRA Model Path")
            n_images = gr.Slider(1, 16, value=1, label="Number of Images", info="How many images to generate")
            n_rows = gr.Slider(1, 4, value=1, label="Rows", info="How many rows to arrange images in")
            decoding_batch_size = gr.Slider(1, 8, value=1, label="Decoding Batch Size", info="Batch size for decoding images")
            output = gr.Textbox("out.png", label="Output File", placeholder="Output file name")
            save_raw = gr.Checkbox(False, label="Save Raw Images", info="Save raw images instead of a grid")
            download_hf = gr.Checkbox(False, label="Download HF Models", info="Download Hugging Face models if not present")
            chroma_path = gr.Textbox("./models/chroma/chroma.safetensors", label="Chroma Model Path", placeholder="Path to Chroma model file")
            t5_path = gr.Textbox("./models/t5/text_encoder_2", label="T5 Model Path", placeholder="Path to T5 model directory")
            tokenizer_path = gr.Textbox("./models/t5/tokenizer_2", label="Tokenizer Path", placeholder="Path to tokenizer directory")
            vae_path = gr.Textbox("./models/vae", label="VAE Model Path", placeholder="Path to VAE model directory")
            quantize = gr.Checkbox(False, label="Load Quantized Models", info="Load quantized models for reduced memory usage")
            
    gen_button = gr.Button("Generate", variant="primary")
    
    gr.Markdown("## Output")
    image_output = gr.Image(label="Generated Image", type="pil", elem_id="output_image",height=512)
    gen_button.click(
        fn=generate,
        inputs=[prompt, neg_prompt, width, height, seed, steps, first_n_steps_without_cfg, cfg, n_images, n_rows, decoding_batch_size, output, save_raw, download_hf, chroma_path, t5_path, tokenizer_path, vae_path, quantize, lora_path, random_seed_check],
        outputs=[image_output, seed],
    )

    
# demo = gr.Interface(fn=greet, inputs="textbox", outputs="textbox")

if __name__ == "__main__":
    demo.launch()
