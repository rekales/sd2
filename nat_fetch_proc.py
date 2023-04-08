import os
import pandas as pd
from PIL import Image
from urllib.request import urlopen

df = pd.read_parquet("5b-010-mini.parquet")

gen_list = os.listdir("./generated_images")
nat_list = os.listdir("./natural_images")

for filename in gen_list:
    if filename in nat_list:
        continue

    id = filename[:-4] # on the assumption that every pic is a png
    url = df.query("SAMPLE_ID==" + id).iloc[0]['URL']

    img = None
    try:
        print("donwloading: " + str(url))
        img = Image.open(urlopen(url))
    except:
        ft = open("retrieval_error_ids.txt", "a")
        ft.write(id)
        ft.write("\n")
        ft.close()
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
        ft = open("retrieval_error_ids.txt", "a")
        ft.write(id)
        ft.write("\n")
        ft.close()
        continue