import numpy as np
import os, time
import utils
import matplotlib.pyplot as plt
import PIL
from PIL import Image

img_dir = '../images/raw'
files = os.listdir(img_dir)

for f in files:
    print f
    basewidth = 350
    img = Image.open(os.path.join(img_dir, f))
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    new_width = 255
    new_height = 255
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img = img.crop((left, top, right, bottom))

    img_path = os.path.join('../images/instances',f)
    print img_path
    img.save(img_path)
