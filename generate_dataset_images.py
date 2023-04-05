from diffusers import StableDiffusionPipeline
import pandas # no fuck you, I'm tired of using pyspark.

df = pandas.read_parquet("5b-010-mini.parquet") # also fuck you, you don't know how long it took me to get this sample

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe = pipe.to("cuda")



















sample = df.sample()

prompt = sample.iloc[0]['TEXT']
image = pipe(prompt).images[0]

image.save(str(int(sample.iloc[0]['SAMPLE_ID']))+".png")