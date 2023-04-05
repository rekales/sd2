from diffusers import StableDiffusionPipeline

print("fetching pretrained model...")
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

pipeline.save_pretrained("pretrained/sd-2-1-base")
print("saved on sd-2-1-base")