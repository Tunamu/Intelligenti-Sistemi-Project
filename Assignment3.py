import cv2
import pytesseract
import os
import numpy as np
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

#For recognition only big letters
valid_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

image_folder = "BigCharacterRepository"
output_csv = "big_pixel_counts.csv"

# Input for row and column size
u = int(input("Kaç satıra bölmek istiyorsunuz? "))

# A list for keep all the result values
results = []
# Column header names for csv file
column_names = ["Image", "Letter"]

for row in range(u):
    for col in range(u):
        column_names.append(f"White_{row}_{col}")
        column_names.append(f"Black_{row}_{col}")

# For every image in our folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: {image_name} is not loaded!")
        continue

    # Looking for the letter in the image
    letter = pytesseract.image_to_string(
        image, config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ).strip()

    if letter not in valid_letters:
        print(f"Warning: {image_name} is nat a valid letter.")
        letter = None

    # Looking for width and height of the image
    h, w = image.shape
    cell_h = h // u
    cell_w = w // u

    # For looking white and black pixels
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    row_data = [image_name, letter]

    # For every cell in the image according to our input
    for row in range(u):
        for col in range(u):
            # Take the region in the image file
            cell = binary_image[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]

            # Calculate white and black pixels
            white_pixels = np.count_nonzero(cell == 255)
            black_pixels = np.count_nonzero(cell == 0)

            # Add the results in the list
            row_data.append(white_pixels)
            row_data.append(black_pixels)

    results.append(row_data)

# Save into custom csv file
df = pd.DataFrame(results, columns=column_names)
df.to_csv(output_csv, index=False)

print(f"Progress complete! The results are saved in {output_csv} file.")
