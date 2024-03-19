import gc
import os
import random
import types
import gradio as gr
import numpy as np
import torch
from typing import List

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from pipelines.pipeline_stable_cascade import StableCascadeDecoderPipelineV2
from pipelines.pipeline_stable_cascade_prior import StableCascadePriorPipelineV2
from models.unets.unet_stable_cascade import StableCascadeUNet
from diffusers.utils import logging
from typing import Optional, List


import os
import datetime
import argparse  # Import the argparse library

# Set up argument parser
parser = argparse.ArgumentParser(description="Gradio interface for text-to-image generation with optional features.")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--torch_compile", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--fp16", action="store_true", help="fp16")
parser.add_argument("--fp8", action="store_true", help="fp8")
parser.add_argument("--lite", action="store_true", help="Uses Lite unet")
logger = logging.get_logger(__name__)

# Parse arguments
args = parser.parse_args()
share = args.share
ENABLE_CPU_OFFLOAD = args.lowvram  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
USE_TORCH_COMPILE = args.torch_compile  # Use the offload argument to toggle ENABLE_CPU_OFFLOAD
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False

dtype = torch.bfloat16
if(args.fp16):
    dtype = torch.float16

dtypeFP8 = dtype
if(args.fp8):
    dtypeFP8 = torch.float8_e5m2

lite = "_lite" if args.lite else ""

print(f"used dtype {dtype}")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
DESCRIPTION = "<p style=\"font-size:14px\">Stable Cascade Modified By SECourses - Unofficial demo for <a href='https://huggingface.co/stabilityai/stable-cascade' target='_blank'>Stable Casacade</a>, a new high resolution text-to-image model by Stability AI, built on the Würstchen architecture.<br/> Some tips: Higher batch size working great with fast speed and not much VRAM usage - Not all resolutions working e.g. 1920x1080 fails but 1920x1152 works<br/>Supports high resolutions very well such as 1536x1536</p>"
if not torch.cuda.is_available():
    DESCRIPTION += "<br/><p>Running on CPU 🥶</p>"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 4096

model_id = "stabilityai/stable-cascade-prior"
model_decoder_id = "stabilityai/stable-cascade"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed
def encode_prompt(
    self,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance: bool = True,
    prompt:str=None,
    negative_prompt:Optional[str]=None,    
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds_pooled: Optional[torch.FloatTensor] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    
    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
            attention_mask = attention_mask[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True
        )
        prompt_embeds = text_encoder_output.hidden_states[-1]        
        if prompt_embeds_pooled is None:
            prompt_embeds_pooled = text_encoder_output.text_embeds.unsqueeze(1)            

    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
    prompt_embeds_pooled = prompt_embeds_pooled.to(dtype=self.text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    prompt_embeds_pooled = prompt_embeds_pooled.repeat_interleave(num_images_per_prompt, dim=0)

    if negative_prompt_embeds is None and do_classifier_free_guidance:
        negative_prompt = negative_prompt or ""

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        
        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds_text_encoder_output = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=uncond_input.attention_mask.to(device),
            output_hidden_states=True,
        )

        negative_prompt_embeds = negative_prompt_embeds_text_encoder_output.hidden_states[-1]        
        negative_prompt_embeds_pooled = negative_prompt_embeds_text_encoder_output.text_embeds.unsqueeze(1)        
    
    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        seq_len = negative_prompt_embeds_pooled.shape[1]
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.to(
            dtype=self.text_encoder.dtype, device=device
        )
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds_pooled = negative_prompt_embeds_pooled.view(
            batch_size * num_images_per_prompt, seq_len, -1            
        )
        # done duplicates

    return prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds, negative_prompt_embeds_pooled
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
    
    if torch.cuda.is_available():
        pipe_encode = types.SimpleNamespace()
        pipe_encode.unet = StableCascadeUNet.from_pretrained(
            model_id, subfolder=fr"prior{lite}").to(device, dtypeFP8)

        pipe_encode.tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder='tokenizer',
        )

        text_encoder_param = {
                'pretrained_model_name_or_path': model_id,
                'subfolder': 'text_encoder',
                'use_safetensors': True,
                'torch_dtype':dtype,
        }
        if dtype == torch.bfloat16:
            text_encoder_param['variant'] = 'bf16'
        pipe_encode.text_encoder = CLIPTextModelWithProjection.from_pretrained(**text_encoder_param).to(device)
      
        text_encoder_param = {
                'pretrained_model_name_or_path': model_id,
                'subfolder': 'text_encoder',
                'use_safetensors': True,
                'torch_dtype': dtype,
        }
        pipeline_param = {
            'pretrained_model_name_or_path': model_id,
            'use_safetensors': True,
            'torch_dtype': dtype,
            'tokenizer': None,
            'text_encoder': None,             
            'prior': pipe_encode.unet,           
        }
        if dtype == torch.bfloat16:
            pipeline_param['variant'] = 'bf16'
        prior_pipeline = StableCascadePriorPipelineV2.from_pretrained(**pipeline_param).to(device)
        
        
        with torch.no_grad():                
            embeddings = encode_prompt(pipe_encode,
                                        device=device,                                         
                                        num_images_per_prompt=batch_size_per_prompt,
                                        prompt=prompt,            
                                        negative_prompt = negative_prompt)            
        
        pipe_decoder_unet = StableCascadeUNet.from_pretrained(
            model_decoder_id, subfolder=fr"decoder{lite}").to(device, dtypeFP8)
        
        pipeline_decoder_param = {
            'pretrained_model_name_or_path': model_decoder_id,
            'use_safetensors': True,
            'torch_dtype': dtype,
            'tokenizer': None,
            'text_encoder': None,             
            'decoder': pipe_decoder_unet,
        }        
        decoder_pipeline = StableCascadeDecoderPipelineV2.from_pretrained(**pipeline_decoder_param).to(device)
       
        del pipe_encode.tokenizer, pipe_encode.text_encoder, pipe_encode.unet, pipe_decoder_unet
        gc.collect()
        torch.cuda.empty_cache()
       
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
       
       
        images = []  # Initialize an empty list to collect generated images
        original_seed = seed  # Store the original seed value
        for i in range(number_of_images_per_prompt):
            if i > 0:  # Update seed for subsequent iterations
                seed = random.randint(0, MAX_SEED)
            generator = torch.Generator().manual_seed(seed)
            with torch.cuda.amp.autocast(dtype=dtype):
                prior_output = prior_pipeline(
                    prompt_embeds=embeddings[0],
                    prompt_embeds_pooled=embeddings[1],
                    negative_prompt_embeds=embeddings[2],
                    negative_prompt_embeds_pooled=embeddings[3],      
                    height=height,
                    width=width,                                                    
                    guidance_scale=prior_guidance_scale,
                    num_images_per_prompt=batch_size_per_prompt,
                    generator=generator,
                    dtype=dtype        
                )                

                decoder_output = decoder_pipeline(
                    image_embeddings=prior_output.image_embeddings,
                    prompt_embeds=embeddings[0],
                    prompt_embeds_pooled=embeddings[1],
                    negative_prompt_embeds=embeddings[2],
                    negative_prompt_embeds_pooled=embeddings[3],      
                    num_inference_steps=decoder_num_inference_steps,
                    guidance_scale=decoder_guidance_scale,                    
                    generator=generator,
                    dtype=dtype,
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