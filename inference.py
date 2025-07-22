import sys
import mlx.core as mx
from chroma import ChromaPipeline
import numpy as np
from numpy.random import Generator, Philox, SeedSequence
import secrets
from PIL import Image


def main(prompt):
    latent_size = (64, 64)

    chroma = ChromaPipeline(
        "chroma",
        download_hf=False,
        chroma_filepath="/Users/unauthorized/Downloads/models/chroma-unlocked-v46.safetensors",
        t5_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/text_encoder_2",
        tokenizer_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21/tokenizer_2",
        vae_filepath="/Users/unauthorized/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    )

    size = 0x100000000
    entropy = f"0x{secrets.randbits(128):x}"  # good entropy
    rndmc = Generator(Philox(SeedSequence(int(entropy, 16))))
    seed = int(rndmc.integers(0, size))

    latent_generator = chroma.generate_latents(
        text=prompt,
        neg_text="",
        num_steps=28,
        seed=seed,
        latent_size=latent_size,
    )

    conditioning = next(latent_generator)
    (
        x_T,  # The initial noise
        x_positions,  # The integer positions used for image positional encoding
        t5_conditioning,  # The T5 features from the text prompt
        t5_positions,  # Integer positions for text (normally all 0s)
        neg_txt,
        neg_txt_ids,
    ) = conditioning

    mx.eval(conditioning)

    for x_t in latent_generator:
        mx.eval(x_t)

    px_data = chroma.decode(x_t, latent_size=latent_size)
    suffix = ".png"

    im = Image.fromarray(np.array(x[i]))
    im.save(".".join([f"image{seed}", str(i), suffix]))

    im.show()


if __name__ == "__main__":
    main(sys.argv[0])
