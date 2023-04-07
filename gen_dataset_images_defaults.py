from diffusers import StableDiffusionPipeline
import torch
import pandas

df = pandas.read_parquet("5b-010-mini.parquet")

pipe = StableDiffusionPipeline.from_pretrained("./pretrained/ds-2-1-base-half", revision="fp16", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

for i in range(1000):
  sample = df.sample()
  prompt = sample.iloc[0]['TEXT']
  image = pipe(
      prompt, 
      guidance_scale=7.5, 
      num_inference_steps=40, 
      ).images[0]

  image.save("generated_images/" + str(int(sample.iloc[0]['SAMPLE_ID']))+".png")