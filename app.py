import gc
import os
import random
import gradio as gr
import numpy as np
import torch

from pipelines.pipeline_common import quantize_4bit, torch_gc
from pipelines.pipeline_stable_cascade import StableCascadeDecoderPipelineV2
from pipelines.pipeline_stable_cascade_prior import StableCascadePriorPipelineV2
from models.unets.unet_stable_cascade import StableCascadeUNet
from diffusers.utils import logging

import os
import datetime
import argparse  # Import the argparse library

# Set up argument parser
parser = argparse.ArgumentParser(description="Gradio interface for text-to-image generation with optional features.")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--torch_compile", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--fp16", action="store_true", help="Load models in fp16.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"], help="Quantization mode for optimization memory consumption")
parser.add_argument("--lite", action="store_true", help="Uses Lite unet")
logger = logging.get_logger(__name__)

# Parse arguments
args = parser.parse_args()
share = args.share

load_mode = args.load_mode
ENABLE_CPU_OFFLOAD = args.lowvram  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
USE_TORCH_COMPILE = args.torch_compile  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

dtypeQuantize = dtype
if(load_mode in ('8bit', '4bit')):
    dtypeQuantize = torch.float8_e4m3fn

lite = "_lite" if args.lite else ""

print(f"used dtype {dtypeQuantize}")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DESCRIPTION = "<p style=\"font-size:14px\">Stable Cascade Modified By SECourses - Unofficial demo for <a href='https://huggingface.co/stabilityai/stable-cascade' target='_blank'>Stable Casacade</a>, a new high resolution text-to-image model by Stability AI, built on the Würstchen architecture.<br/> Some tips: Higher batch size working great with fast speed and not much VRAM usage - Not all resolutions working e.g. 1920x1080 fails but 1920x1152 works<br/>Supports high resolutions very well such as 1536x1536</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "<br/><p>Running on CPU 🥶</p>"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 4096

model_id = "stabilityai/stable-cascade-prior"
model_decoder_id = "stabilityai/stable-cascade"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe_prior_unet = None
prior_pipeline = None
pipe_decoder_unet = None
decoder_pipeline = None

def restart_cpu_offload():
    if load_mode != '4bit' :
        prior_pipeline.disable_xformers_memory_efficient_attention()
        decoder_pipeline.disable_xformers_memory_efficient_attention()               
    from pipelines.pipeline_common import optionally_disable_offloading
    optionally_disable_offloading(prior_pipeline)
    optionally_disable_offloading(decoder_pipeline)
    gc.collect()
    torch.cuda.empty_cache()
    prior_pipeline.enable_model_cpu_offload()
    decoder_pipeline.enable_model_cpu_offload()
    if load_mode != '4bit' :
        prior_pipeline.enable_xformers_memory_efficient_attention()
        decoder_pipeline.enable_xformers_memory_efficient_attention()

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
    global pipe_prior_unet, prior_pipeline, pipe_decoder_unet, decoder_pipeline, need_restart_cpu_offloading
    if torch.cuda.is_available():        
        need_restart_cpu_offloading = False
        if prior_pipeline == None:
            pipe_prior_unet = StableCascadeUNet.from_pretrained(
            model_id, subfolder=fr"prior{lite}").to(device, dtypeQuantize)
            
            if load_mode == '4bit':
                quantize_4bit(pipe_prior_unet)

            pipeline_param = {
                'pretrained_model_name_or_path': model_id,
                'use_safetensors': True,
                'torch_dtype': dtype,   
                'prior':pipe_prior_unet
            }
            prior_pipeline = StableCascadePriorPipelineV2.from_pretrained(**pipeline_param).to(device)
            if load_mode == '4bit':
                if prior_pipeline.text_encoder is not None:
                    quantize_4bit(prior_pipeline.text_encoder)
              
            if load_mode != '4bit' :
                prior_pipeline.enable_xformers_memory_efficient_attention()           
        else:
            if ENABLE_CPU_OFFLOAD:
                need_restart_cpu_offloading =True
        torch_gc()
        if decoder_pipeline == None:
            pipe_decoder_unet = StableCascadeUNet.from_pretrained(
            model_decoder_id, subfolder=fr"decoder{lite}").to(device, dtypeQuantize)
                
            if load_mode == '4bit':
                quantize_4bit(pipe_decoder_unet)

            pipeline_decoder_param = {
                'pretrained_model_name_or_path': model_decoder_id,
                'use_safetensors': True,
                'torch_dtype': dtype,
                'decoder': pipe_decoder_unet,
            }        
            decoder_pipeline = StableCascadeDecoderPipelineV2.from_pretrained(**pipeline_decoder_param,).to(device)
            
            if load_mode == '4bit':
                if decoder_pipeline.text_encoder is not None:
                    quantize_4bit(decoder_pipeline.text_encoder)

            if load_mode != '4bit' :
                decoder_pipeline.enable_xformers_memory_efficient_attention()
        
        else:
            if ENABLE_CPU_OFFLOAD:
                need_restart_cpu_offloading=True

        torch_gc()
        
        if need_restart_cpu_offloading:
            restart_cpu_offload()
        elif ENABLE_CPU_OFFLOAD:
            prior_pipeline.enable_model_cpu_offload()
            decoder_pipeline.enable_model_cpu_offload()
        
        if USE_TORCH_COMPILE:
            prior_pipeline.prior = torch.compile(prior_pipeline.prior, mode="reduce-overhead", fullgraph=True)
            decoder_pipeline.decoder = torch.compile(decoder_pipeline.decoder, mode="max-autotune", fullgraph=True)
       
       
        images = []  # Initialize an empty list to collect generated images
        original_seed = seed  # Store the original seed value
        for i in range(number_of_images_per_prompt):
            if i > 0:  # Update seed for subsequent iterations
                seed = random.randint(0, MAX_SEED)
            generator = torch.Generator().manual_seed(seed)
            with torch.cuda.amp.autocast(dtype=dtype):
                prior_output = prior_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,             
                    num_inference_steps=prior_num_inference_steps,
                    height=height,
                    width=width,                                                    
                    guidance_scale=prior_guidance_scale,
                    num_images_per_prompt=batch_size_per_prompt,
                    generator=generator,
                    dtype=dtype,
                    device=device,      
                )                

                decoder_output = decoder_pipeline(
                    image_embeddings=prior_output.image_embeddings,
                    prompt=prompt,
                    negative_prompt=negative_prompt,                    
                    num_inference_steps=decoder_num_inference_steps,
                    guidance_scale=decoder_guidance_scale,                    
                    generator=generator,
                    dtype=dtype,
                    device=device,
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
    else:
        prior_pipeline = None
        decoder_pipeline = None
        return


with gr.Blocks() as app:
    gr.Markdown("## V8 Stable Cascade by SECourses : 1-Click Installers Latest Version On : https://www.patreon.com/posts/98410661")
    gr.Markdown("[Stable Cascade](https://stability.ai/news/introducing-stable-cascade) is the latest model of Stability AI based on Würstchen architecture")
    gr.Markdown("Stable Cascade is compatible with GPUs having as little as 5 GB of memory and can generate high-quality images at resolutions even at 1536x1536 pixels. It supports resolution adjustments in 128-pixel steps, e.g. 1024x1024 or 1152x1024 or 1152x896")
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