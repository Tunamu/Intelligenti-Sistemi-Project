import numpy as np 
import cv2 as sv
from PIL import Image

import pytesseract

image = cv.imread('./data/IMG_5239.jpg')
assert image is not None ,"file could not be read, check with os.path.exists()"
ball = img[280:340, 330:390]
print(ball)
#print(pytesseract.image_to_string(Image.open('IMG_5239.jpg')))
#My Datalist : IMG_5239.jpg - IMG_5254.jpg
