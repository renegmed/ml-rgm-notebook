'''
Transform image into vector (single dim array)
'''
from PIL import Image
import numpy as np

img = Image.open("testSet/testSet/img_1.jpg")
arr = np.array(img)

arr = arr.flatten()

print(arr)
