import os
import numpy as np
from PIL import Image

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("exists")
    return

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)