import cv2
import pytesseract
import os
import numpy as np
import pandas as pd

# Geçerli harfler
valid_letters = "ABCDEFGHIJKLMNOPRSTUVXYZ"

# Tesseract path (Windows kullanıyorsan burayı kendi kurulumuna göre ayarla)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Resimlerin bulunduğu klasör
image_folder = "bigCharacters"
output_csv = "pixel_counts.csv"

# Satır sayısını kullanıcıdan al
u = int(input("Kaç satıra bölmek istiyorsunuz? "))

# Kaç sütuna böleceğimiz belli: 3
cols = 3

# Sonuçları saklamak için liste
results = []
column_names = ["Image", "Letter"]  # İlk iki sütun: Görüntü adı ve harf

# Dinamik olarak kolon isimleri oluştur
for row in range(u):
    for col in range(cols):
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

    # 1. Dosya adından harfi al (ve geçerli mi kontrol et)
    letter = image_name[0].upper() if image_name[0].isalpha() else None
    if letter not in valid_letters:
        letter = None  # Dosya adındaki harf geçerli değilse OCR kullan

    # 2. OCR ile harfi belirleme (Eğer dosya adında harf yoksa)
    if not letter:
        letter = pytesseract.image_to_string(
            image, config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPRSTUVXYZ"
        ).strip()

    # 3. Eğer harf geçerli değilse işlemi atla
    if letter not in valid_letters:
        print(f"Uyarı: {image_name} geçerli bir harf içermiyor, atlanıyor.")
        continue

    # Resmin yüksekliği ve genişliği
    h, w = image.shape
    cell_h = h // u  # Her bir hücrenin yüksekliği
    cell_w = w // cols  # Her bir hücrenin genişliği

    # Görüntüyü binary hale getir (Siyah-beyaz ayrımı için)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Tek satırda tüm bilgileri tutacak liste
    row_data = [image_name, letter]

    # Hücreleri işle ve tek satıra ekle
    for row in range(u):
        for col in range(cols):
            # Bölgeyi al
            cell = binary_image[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]

            # Siyah ve beyaz piksel sayısını hesapla
            white_pixels = np.count_nonzero(cell == 255)
            black_pixels = np.count_nonzero(cell == 0)

            # Sonuçları listeye ekle (tek satır formatında)
            row_data.append(white_pixels)
            row_data.append(black_pixels)

    # Sonuçlar listesine ekle
    results.append(row_data)

# CSV'ye yaz
df = pd.DataFrame(results, columns=column_names)
df.to_csv(output_csv, index=False)

print(f"İşlem tamamlandı! Sonuçlar {output_csv} dosyasına kaydedildi.")
