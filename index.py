import numpy as np 
import cv2 as cv
from PIL import Image

import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


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
text = pytesseract.image_to_boxes(image)
cv.imshow('image',image)
print(text)

# Kullanıcının bir tuşa basmasını bekle
cv.waitKey(0)

# Tüm OpenCV pencerelerini kapat
cv.destroyAllWindows()


#My Datalist : IMG_5239.jpg - IMG_5254.jpg