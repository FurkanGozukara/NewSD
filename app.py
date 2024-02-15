import os
import random
import gradio as gr
import numpy as np
import PIL.Image
import torch
from typing import List
from diffusers.utils import numpy_to_pil
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

import os
import datetime
import json
import io
import argparse  # Import the argparse library

# Set up argument parser
parser = argparse.ArgumentParser(description="Gradio interface for text-to-image generation with optional features.")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--torch_compile", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--fp16", action="store_true", help="fp16")

# Parse arguments
args = parser.parse_args()
share = args.share
ENABLE_CPU_OFFLOAD = args.lowvram  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
USE_TORCH_COMPILE = args.torch_compile  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
torch.backends.cuda.matmul.allow_tf32 = True

dtype = torch.bfloat16
if(args.fp16):
    dtype = torch.float16

print(f"used dtype {dtype}")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DESCRIPTION = "<p style=\"font-size:14px\">Stable Cascade Modified By SECourses - Unofficial demo for <a href='https://huggingface.co/stabilityai/stable-cascade' target='_blank'>Stable Casacade</a>, a new high resolution text-to-image model by Stability AI, built on the Würstchen architecture.<br/> Some tips: Higher batch size working great with fast speed and not much VRAM usage - Not all resolutions working e.g. 1920x1080 fails but 1920x1152 works<br/>Supports high resolutions very well such as 1536x1536</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "<br/><p>Running on CPU 🥶</p>"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 4096


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    
    prior_pipeline = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", 
                                                                torch_dtype=dtype).to(device)     
    decoder_pipeline = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=dtype).to(device)
    prior_pipeline.enable_xformers_memory_efficient_attention()
    decoder_pipeline.enable_xformers_memory_efficient_attention()

    if ENABLE_CPU_OFFLOAD:
        prior_pipeline.enable_model_cpu_offload()
        decoder_pipeline.enable_model_cpu_offload()
    # else:
    #     prior_pipeline.to(device)
    #     decoder_pipeline.to(device)

    if USE_TORCH_COMPILE:
        prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
        decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="max-autotune", fullgraph=True)
    
else:
    prior_pipeline = None
    decoder_pipeline = None


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 30,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 12,
    decoder_guidance_scale: float = 0.0,
    batch_size_per_prompt: int = 2,
    number_of_images_per_prompt: int = 1,  # New parameter
):
    images = []  # Initialize an empty list to collect generated images
    original_seed = seed  # Store the original seed value
    for i in range(number_of_images_per_prompt):
        if i > 0:  # Update seed for subsequent iterations
            seed = random.randint(0, MAX_SEED)
        generator = torch.Generator().manual_seed(seed)
        with torch.cuda.amp.autocast(dtype=dtype):
            prior_output = prior_pipeline(
                prompt=prompt,
                height=height,
                width=width,                                
                negative_prompt=negative_prompt,
                guidance_scale=prior_guidance_scale,
                num_images_per_prompt=batch_size_per_prompt,
                generator=generator                
            )

            decoder_output = decoder_pipeline(
                image_embeddings=prior_output.image_embeddings,
                prompt=prompt,
                num_inference_steps=decoder_num_inference_steps,
                guidance_scale=decoder_guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                output_type="pil",
            ).images

        # Append generated images to the images list
        images.extend(decoder_output)

        # Optionally, save each image
        output_folder = 'outputs'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for image in decoder_output:
            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
            image_filename = f"{output_folder}/{timestamp}.png"
            image.save(image_filename)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Return the list of generated images
    return images

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            prompt = gr.Text(
                label="Prompt",
                placeholder="Enter your prompt",
            )
            run_button = gr.Button("Generate")
            
            # Advanced options now directly visible
            negative_prompt = gr.Text(
                label="Negative prompt",
                placeholder="Enter a Negative Prompt",
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(
                        label="Width",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=128,
                        value=1024,
                    )
                with gr.Column():
                    height = gr.Slider(
                        label="Height",
                        minimum=512,
                        maximum=MAX_IMAGE_SIZE,
                        step=128,
                        value=1024,
                    )
            with gr.Row():
                with gr.Column():
                    batch_size_per_prompt = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=1,
                    )
                with gr.Column():
                    number_of_images_per_prompt = gr.Slider(
                        label="Number Of Images To Generate",
                        minimum=1,
                        maximum=9999999,
                        step=1,
                        value=1,
                    )
            with gr.Row():
                with gr.Column():
                    prior_guidance_scale = gr.Slider(
                        label="Prior Guidance Scale (CFG)",
                        minimum=0,
                        maximum=20,
                        step=0.1,
                        value=4.0,
                    )
                with gr.Column():
                    decoder_guidance_scale = gr.Slider(
                        label="Decoder Guidance Scale (CFG)",
                        minimum=0,
                        maximum=20,
                        step=0.1,
                        value=0.0,
                    )
            with gr.Row():
                with gr.Column():
                    prior_num_inference_steps = gr.Slider(
                        label="Prior Inference Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=30,
                    )
                with gr.Column():
                    decoder_num_inference_steps = gr.Slider(
                        label="Decoder Inference Steps",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=20,
                    )
            
        with gr.Column():
            result = gr.Gallery(label="Result", show_label=False, height=768)

    inputs = [
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            prior_num_inference_steps,
            # prior_timesteps,
            prior_guidance_scale,
            decoder_num_inference_steps,
            # decoder_timesteps,
            decoder_guidance_scale,
            batch_size_per_prompt,
            number_of_images_per_prompt
    ]
    gr.on(
        triggers=[prompt.submit, negative_prompt.submit, run_button.click],
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=inputs,
        outputs=result,
        api_name="run",
    )
		
if __name__ == "__main__":
    app.queue().launch(share=share,inbrowser=True)