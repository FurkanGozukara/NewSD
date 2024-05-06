import gc
import os
import random
import sys
import platform
sys.path.append("./")
import gradio as gr
import numpy as np
import torch
from PIL import Image, PngImagePlugin
import datetime
import csv
from pipelines.pipeline_common import quantize_4bit, torch_gc
from pipelines.pipeline_stable_cascade import StableCascadeDecoderPipelineV2
from pipelines.pipeline_stable_cascade_prior import StableCascadePriorPipelineV2
from models.unets.unet_stable_cascade import StableCascadeUNet
from diffusers.utils import logging

import argparse

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
ENABLE_CPU_OFFLOAD = args.lowvram
USE_TORCH_COMPILE = args.torch_compile
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

dtypeQuantize = dtype
if load_mode in ('8bit', '4bit'):
    dtypeQuantize = torch.float8_e4m3fn

lite = "_lite" if args.lite else ""

print(f"used dtype {dtypeQuantize}")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

if not torch.cuda.is_available():
    print("Running on CPU 🥶")

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 4096

model_id = "stabilityai/stable-cascade-prior"
model_decoder_id = "stabilityai/stable-cascade"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe_prior_unet = None
prior_pipeline = None
pipe_decoder_unet = None
decoder_pipeline = None

def load_styles():
    styles = {"No Style": ("", "")}
    try:
        with open('styles.csv', mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) == 3:
                    styles[row[0]] = (row[1], row[2])
    except Exception as e:
        print(f"Failed to load styles from CSV: {e}")
    return styles

styles = load_styles()

def restart_cpu_offload():
    if load_mode != '4bit':
        prior_pipeline.disable_xformers_memory_efficient_attention()
        decoder_pipeline.disable_xformers_memory_efficient_attention()
    from pipelines.pipeline_common import optionally_disable_offloading
    optionally_disable_offloading(prior_pipeline)
    optionally_disable_offloading(decoder_pipeline)
    gc.collect()
    torch.cuda.empty_cache()
    prior_pipeline.enable_model_cpu_offload()
    decoder_pipeline.enable_model_cpu_offload()
    if load_mode != '4bit':
        prior_pipeline.enable_xformers_memory_efficient_attention()
        decoder_pipeline.enable_xformers_memory_efficient_attention()

def read_image_metadata(image_path):
    if image_path is None or not os.path.exists(image_path):
        return "File does not exist or path is None."
    
    
    last_modified_timestamp = os.path.getmtime(image_path)

    last_modified_date = datetime.datetime.fromtimestamp(last_modified_timestamp).strftime('%d %B %Y, %H:%M %p - UTC')
    with Image.open(image_path) as img:
        metadata = img.info
        metadata_str = f"Last Modified Date: {last_modified_date}\n"
        for key, value in metadata.items():
            metadata_str += f"{key}: {value}\n"
            
    return metadata_str

def save_image_with_metadata(image, filename, metadata):
    meta_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        meta_info.add_text(key, str(value))
    image.save(filename, "PNG", pnginfo=meta_info)

def set_metadata_settings(image_path, style_dropdown):
    if image_path is None:
        return (gr.update(),) * 11  # Return a tuple of 11 gr.update() calls
    
    with Image.open(image_path) as img:
        metadata = img.info
        prompt = metadata.get("Prompt", "")
        negative_prompt = metadata.get("Negative Prompt", "")
        style = metadata.get("Style", "No Style")
        seed = int(metadata.get("Seed", "0"))
        width = int(metadata.get("Width", "1024"))
        height = int(metadata.get("Height", "1024"))
        prior_guidance_scale = float(metadata.get("Prior Guidance Scale", "4.0"))
        decoder_guidance_scale = float(metadata.get("Decoder Guidance Scale", "0.0"))
        prior_num_inference_steps = int(metadata.get("Prior Inference Steps", "30"))
        decoder_num_inference_steps = int(metadata.get("Decoder Inference Steps", "20"))
        batch_size_per_prompt = int(metadata.get("Batch Size", "1"))
        number_of_images_per_prompt = int(metadata.get("Number Of Images To Generate", "1"))

    # Construct the updates list with gr.update calls for each setting
    updates = [
        gr.update(value=prompt),
        gr.update(value=negative_prompt),
        gr.update(value=style),
        gr.update(value=seed),
        gr.update(value=width),
        gr.update(value=height),
        gr.update(value=prior_guidance_scale),
        gr.update(value=decoder_guidance_scale),
        gr.update(value=prior_num_inference_steps),
        gr.update(value=decoder_num_inference_steps),
        gr.update(value=batch_size_per_prompt),
        gr.update(value=number_of_images_per_prompt)
    ]
    
    return tuple(updates)

def generate(
    prompt: str,
    negative_prompt: str = "",
    style: str = "No Style",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    prior_num_inference_steps: int = 30,
    prior_guidance_scale: float = 4.0,
    decoder_num_inference_steps: int = 12,
    decoder_guidance_scale: float = 0.0,
    batch_size_per_prompt: int = 2,
    number_of_images_per_prompt: int = 1,
    randomize_seed_ck: bool = False,
    loop_styles_ck: bool = False  # New parameter to handle looping through styles
):
    global pipe_prior_unet, prior_pipeline, pipe_decoder_unet, decoder_pipeline, need_restart_cpu_offloading
    if torch.cuda.is_available():        
        need_restart_cpu_offloading = False
        if prior_pipeline is None:
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
        if decoder_pipeline is None:
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
            decoder_pipeline = StableCascadeDecoderPipelineV2.from_pretrained(**pipeline_decoder_param).to(device)
            
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
        original_prompt = prompt
        original_neg_prompt = negative_prompt

        # Use all styles if loop_styles_ck is True, otherwise use only the selected style
        selected_styles = styles if loop_styles_ck else [(style, "", "")]
        total_images = len(selected_styles) * number_of_images_per_prompt
        image_counter = 1

        for style_name in selected_styles:
            get_name = style_name[0]
            if(len(get_name) < 2):
                get_name = style_name
            style_prompt, style_negative_prompt = styles.get(get_name, ("", ""))
    
            # Replace placeholders in the style prompt
            prompt = style_prompt.replace("{prompt}", original_prompt) if style_prompt else original_prompt
            negative_prompt = style_negative_prompt if style_negative_prompt else original_neg_prompt

            print(f"\nFinal Prompt: {prompt}")       
            print(f"Final Negative Prompt: {negative_prompt}\n")     

            for i in range(number_of_images_per_prompt):
                if randomize_seed_ck or i > 0:  # Update seed if randomize is checked or for subsequent images
                    seed = random.randint(0, MAX_SEED)

                print(f"Image {image_counter}/{total_images} Being Generated")
                image_counter=image_counter+1
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
                
                    # Prepare metadata
                    metadata = {
                        "Prompt": original_prompt,
                        "Negative Prompt": original_neg_prompt,
                        "Style":style,
                        "Seed": seed,
                        "Width": width,
                        "Height": height,
                        "Prior Guidance Scale": prior_guidance_scale,
                        "Decoder Guidance Scale": decoder_guidance_scale,
                        "Prior Inference Steps": prior_num_inference_steps,
                        "Decoder Inference Steps": decoder_num_inference_steps
                    }
                
                    # Save image with metadata
                    save_image_with_metadata(image, image_filename, metadata)
                
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Return the list of generated images
        return images, seed
    else:
        prior_pipeline = None
        decoder_pipeline = None
        return

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

# Modify the existing Blocks setup to add a dropdown for styles
with gr.Blocks() as app:
    gr.Markdown(""" ### Stable Cascade V10 by SECourses : 1-Click Installers Latest Version On : https://www.patreon.com/posts/98410661 
    ### [Stable Cascade](https://stability.ai/news/introducing-stable-cascade) is the latest model of Stability AI based on Würstchen architecture 
    ### Stable Cascade is compatible with GPUs having as little as 5 GB of memory and can generate high-quality images at resolutions even at 1536x1536 pixels. It supports resolution adjustments in 128-pixel steps, e.g. 1024x1024 or 1152x1024 or 1152x896""")

    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                # Main settings column
                with gr.Row():
                    prompt = gr.Text(label="Prompt", placeholder="Enter your prompt")
                with gr.Row():
                    negative_prompt = gr.Text(label="Negative prompt", placeholder="Enter a Negative Prompt")
                with gr.Row():
                    style_dropdown = gr.Dropdown(label="Style", choices=list(styles.keys()), value="No Style")
                    loop_styles_ck = gr.Checkbox(label="Loop All Styles", value=False)  # New Checkbox for looping through all styles
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed_ck = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=512, maximum=MAX_IMAGE_SIZE, step=128, value=1024)
                    height = gr.Slider(label="Height", minimum=512, maximum=MAX_IMAGE_SIZE, step=128, value=1024)
                with gr.Row():
                    batch_size_per_prompt = gr.Slider(label="Batch Size", minimum=1, maximum=20, step=1, value=1)
                    number_of_images_per_prompt = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=9999999, step=1, value=1)
                with gr.Row():
                    prior_guidance_scale = gr.Slider(label="Prior Guidance Scale (CFG)", minimum=0, maximum=20, step=0.1, value=4.0)
                    decoder_guidance_scale = gr.Slider(label="Decoder Guidance Scale (CFG)", minimum=0, maximum=20, step=0.1, value=0.0)
                with gr.Row():
                    prior_num_inference_steps = gr.Slider(label="Prior Inference Steps", minimum=1, maximum=100, step=1, value=30)
                    decoder_num_inference_steps = gr.Slider(label="Decoder Inference Steps", minimum=1, maximum=100, step=1, value=20)
                with gr.Row():
                    run_button = gr.Button("Generate")
                with gr.Row():
                    btn_open_outputs = gr.Button("Open Outputs Folder (Works on Windows & Desktop Linux)")
                    btn_open_outputs.click(fn=open_folder)
            with gr.Column():
                # Output and additional settings column
                result = gr.Gallery(label="Result", show_label=False, height=768)

            run_button.click(fn=generate, inputs=[
                prompt, negative_prompt, style_dropdown, seed, width, height,
                prior_num_inference_steps, prior_guidance_scale,
                decoder_num_inference_steps, decoder_guidance_scale,
                batch_size_per_prompt, number_of_images_per_prompt, randomize_seed_ck, loop_styles_ck
            ], outputs=[result, seed])

    with gr.Tab("Image Metadata"):
        with gr.Row():
            set_metadata_button = gr.Button("Load & Set Metadata Settings")
        with gr.Row():
            with gr.Column():
                metadata_image_input = gr.Image(type="filepath", label="Upload Image")
            with gr.Column():
                metadata_output = gr.Textbox(label="Image Metadata", lines=25, max_lines=50)

        metadata_image_input.change(fn=read_image_metadata, inputs=[metadata_image_input], outputs=[metadata_output])
        set_metadata_button.click(fn=set_metadata_settings, inputs=[metadata_image_input, style_dropdown], outputs=[
            prompt, negative_prompt, style_dropdown, seed, width, height,
            prior_guidance_scale, decoder_guidance_scale, prior_num_inference_steps, decoder_num_inference_steps,
            batch_size_per_prompt, number_of_images_per_prompt,
        ])

if __name__ == "__main__":
    app.launch(share=share, inbrowser=True)