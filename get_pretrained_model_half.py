from diffusers import StableDiffusionPipeline
import torch

print("fetching pretrained model...")
# pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", revision="fp16", torch_dtype=torch.float16)


pipeline.save_pretrained("pretrained/sd-2-1-base-half")
print("saved on pretrained/sd-2-1-base-half")