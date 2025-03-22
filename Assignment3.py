import cv2
import pytesseract
import os
import numpy as np
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Resimlerin bulunduğu klasör
image_folder = "BigCharacterRepository"
output_csv = "big_pixel_counts.csv"

# Satır sayısını kullanıcıdan al
u = int(input("Kaç satıra bölmek istiyorsunuz? "))

# Sonuçları saklamak için liste
results = []
column_names = ["Image", "Letter"]

for row in range(u):
    for col in range(u):
        column_names.append(f"White_{row}_{col}")
        column_names.append(f"Black_{row}_{col}")

# Klasördeki tüm resimleri işle
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # Resmi yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Hata: {image_name} yüklenemedi!")
        continue

    letter = pytesseract.image_to_string(
        image, config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVXYZ"
    ).strip()

    if letter not in valid_letters:
        print(f"Uyarı: {image_name} geçerli bir harf içermiyor, atlanıyor.")
        continue

    # Resmin yüksekliği ve genişliği
    h, w = image.shape
    cell_h = h // u  # Her bir hücrenin yüksekliği
    cell_w = w // u  # Her bir hücrenin genişliği

    # Görüntüyü binary hale getir (Siyah-beyaz ayrımı için)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    row_data = [image_name, letter]

    # Hücreleri işle
    for row in range(u):
        for col in range(u):
            # Bölgeyi al
            cell = binary_image[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]

            # Siyah ve beyaz piksel sayısını hesapla
            white_pixels = np.count_nonzero(cell == 255)
            black_pixels = np.count_nonzero(cell == 0)

            # Sonuçları listeye ekle
            row_data.append(white_pixels)
            row_data.append(black_pixels)

    results.append(row_data)

# CSV'ye yaz
df = pd.DataFrame(results, columns=column_names)
df.to_csv(output_csv, index=False)

print(f"İşlem tamamlandı! Sonuçlar {output_csv} dosyasına kaydedildi.")
