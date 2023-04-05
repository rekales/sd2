from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("./pretrained/ds-2-1-base")
pipe = pipe.to("cuda")

prompt = "abstract realism art fusion"
image = pipe(prompt).images[0]

image.save(f"random_image.png")