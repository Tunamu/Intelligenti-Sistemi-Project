import cv2
import numpy as np
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Dosyanın var olup olmadığını kontrol et
image_path = "./data/IMG_5239.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

# OpenCV ile yükleme
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görsel yüklenemedi. Dosya yolu yanlış veya dosya bozuk: {image_path}")

# Görüntü işleme devam eder...


# Görüntüyü yükle
image_path = "./data/IMG_5239.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Görüntüyü ters çevirme (beyaz arkaplan, siyah yazılar olacak)
_, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

# Kenarları algılamak için konturlar bulma
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tespit edilen dikdörtgenleri saklayalım
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Satır ve sütun bazlı sıralama

# Ortalama hücre genişliği ve yüksekliğini belirle
cell_width = int(np.mean([w for x, y, w, h in boxes]))
cell_height = int(np.mean([h for x, y, w, h in boxes]))

# Sütun sayısını belirle
num_columns = max([x // cell_width for x, y, w, h in boxes]) + 1
num_rows = max([y // cell_height for x, y, w, h in boxes]) + 1

# Sonuçları saklayacağımız 2D array
grid_data = [[] for _ in range(num_columns)]

# Hücreleri OCR ile okumak
for x, y, w, h in sorted(boxes, key=lambda b: (b[0], b[1])):  # Sütun bazlı sıralama
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (cell_width, cell_height))

    # OCR ile karakter tanıma
    char = pytesseract.image_to_boxes(roi, config='--psm 10').strip()

    # Sütuna ekleme (hücre sırasına göre)
    col_index = x // cell_width
    grid_data[col_index].append(char)

# Çıktıyı ekrana yazdır
for i, column in enumerate(grid_data):
    print(f"Sütun {i + 1}: {column}")
