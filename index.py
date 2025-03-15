import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

output_folder = "CharacterRepo"
os.makedirs(output_folder, exist_ok=True)

image_path = "./data/bigCharacter.png"

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Görsel dosyası bulunamadı: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görsel yüklenemedi. Dosya yolu yanlış veya dosya bozuk: {image_path}")

_, thresh = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.imshow("Thresh", thresh)
cv2.imshow("Contours", image_color)

boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Satır ve sütun bazlı sıralama

#for (x, y, w, h) in boxes:
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Image", image)

cell_width = int(np.mean([w for x, y, w, h in boxes]))
cell_height = int(np.mean([h for x, y, w, h in boxes]))
print(cell_width, cell_height)

char_index = 1

for x, y, w, h in sorted(boxes, key=lambda b: (b[0], b[1])):  # Sütun bazlı sıralama
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (cell_width, cell_height))

    char = pytesseract.image_to_boxes(roi, config='--psm 10').strip()

    #TODO
    #if not char or not char.isalnum():
    #    continue

    char_filename = f"{output_folder}/char_{char_index}.png"

    cv2.imwrite(char_filename, roi)

    print(f"Karakter '{char}' kaydedildi: {char_filename}")

    char_index += 1

cv2.waitKey(0)
cv2.destroyAllWindows()