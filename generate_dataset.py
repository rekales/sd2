import os
import pandas as pd
from PIL import Image
from urllib.request import urlopen
from diffusers import StableDiffusionPipeline
import torch

df = pd.read_parquet("5b-010-mini.parquet")
gen_list = os.listdir("./generated_images")

pipe = StableDiffusionPipeline.from_pretrained("./pretrained/ds-2-1-base-half", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

for i in range(10):
    sample = df.sample()
    filename = str(sample.iloc[0]['SAMPLE_ID']) + ".png"

    if filename in gen_list:
        continue

    # Retrieve Image
    url = sample.iloc[0]['URL']
    try:
        print("Downloading: " + str(url))
        img = Image.open(urlopen(url))
    except:
        print("Retrieval Error: " + str(sample.iloc[0]['SAMPLE_ID']))
        continue    
    
    # crop image to 1:1 aspect ratio
    if img.width > img.height:
        left = int((img.height-img.width)/2)
        img = img.crop((left, 0, img.height+left, img.height))
    elif img.height > img.width:
        top = int((img.width-img.height)/2)
        img = img.crop((0, top, img.width, img.width+top))

    # scale image
    img = img.resize((512, 512))
    try:
        img.save("./natural_images/" + filename)
    except:
        print("Saving Error" + str(sample.iloc[0]['SAMPLE_ID']))
        continue


    # Generate Image
    print("Generating: " + sample.iloc[0]['TEXT'])
    prompt = sample.iloc[0]['TEXT']
    image = pipe(
        prompt, 
        guidance_scale=7.5, 
        num_inference_steps=50, 
        ).images[0]
    image.save("generated_images/" + filename)


    gen_list.append(filename)