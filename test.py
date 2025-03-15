import numpy as np 
import cv2 as cv
from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

from matplotlib import pyplot as plt

img = cv.imread('./data/IMG_5254.jpg', cv.IMREAD_GRAYSCALE)  # `<opencv_root>/samples/data/blox.jpg`

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(opening, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

cv.imshow('fast_true.png', img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

img3 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv.imshow('fast_false.png', img3)


#ball = image[10:100, 10:100]
#print(ball)


# Tesseract ile OCR çalıştır
#text = pytesseract.image_to_string(img)
#print(text)


image_path = './data/IMG_5239.jpg'

# OpenCV ile oku
image = cv.imread(image_path)
assert image is not None, "Dosya okunamadı!"

# OpenCV -> RGB formatına çevir
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


# OpenCV görüntüsünü PIL formatına çevirerek Tesseract’a gönder
pil_image = Image.fromarray(image)
ball = image[100:150, 100:150]
#text = pytesseract.image_to_boxes(image)
cv.imshow('image',ball)
#print(text)



# Kullanıcının bir tuşa basmasını bekle
cv.waitKey(0)

# Tüm OpenCV pencerelerini kapat
cv.destroyAllWindows()


#My Datalist : IMG_5239.jpg - IMG_5254.jpg