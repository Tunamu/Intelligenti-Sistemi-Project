import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

output_folder = "CharacterRepo"
os.makedirs(output_folder, exist_ok=True)

image_path = "./data/littleCharacter.png.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görsel yüklenemedi. Dosya yolu yanlış veya dosya bozuk: {image_path}")

# Görüntüyü ters çevir (Beyaz karakterler - Siyah arka plan)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

cv2.imshow("Thresh", otsu_thresh)
cv2.imshow("Characters Only", image)
cv2.imshow("Contours", image_color)

# Bounding box'ları sırala (Önce satır bazlı, sonra sütun bazlı)
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Önce satır bazlı, sonra sütun bazlı sıralama

# Bounding box'ları sırala (Önce satır bazlı, sonra sütun bazlı)
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Önce satır bazlı, sonra sütun bazlı sıralama

char_index = 1

for x, y, w, h in boxes:
    roi = image[y:y + h, x:x + w]

    # OCR kullanarak karakteri oku
    char = pytesseract.image_to_string(roi, config='--psm 10').strip()

    # Eğer karakter boşsa veya alfanumerik değilse kaydetme
    if not char or not char.isalnum():
        print(f"Atlandı: ({x}, {y}, {w}, {h}) -> '{char}'")
        continue

    char_filename = f"{output_folder}/char_{char_index}.png"
    cv2.imwrite(char_filename, roi)

    print(f"Karakter '{char}' kaydedildi: {char_filename}")

    char_index += 1

cv2.waitKey(0)
cv2.destroyAllWindows()