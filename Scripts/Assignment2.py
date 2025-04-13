import cv2
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

output_folder = "LittleCharacterRepository"
os.makedirs(output_folder, exist_ok=True)

image_path = "../data/littleCharacter.png"

# We look any errors because of the image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image path could not find!: {image_path}")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #Open the image with black and white format(GRAYSCALE)

# We look that can we open the image
if image is None:
    raise FileNotFoundError(f"Image does not loaded!: {image_path}")

# I use otsu tresh method to optimise the rendering process
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# We find lines inside of the image to detect box and we use tree method to detect all the boxes
contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# For observing the lines we draw in image
cv2.drawContours(image_color, contours, -1, (0, 255, 0) , 2)

# Draw all the box finding process
cv2.imshow("Thresh", otsu_thresh)
cv2.imshow("Image", image)
cv2.imshow("Contours", image_color)


#Creating boxes in the order of rows
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0])) #Fırst based on row then column

char_index = 1 #For every boxes we add 1 for name that boxes

for x, y, w, h in boxes:
    roi = image[y:y + h, x:x + w]

    # Read character using OCR
    char = pytesseract.image_to_string(roi, config='--psm 10').strip()

    # If the scan of the box is empty or unusable
    if not char or not char.isalnum():
        print(f"Passed: ({x}, {y}, {w}, {h}) -> '{char}'")
        continue

    # Save as unique name into that file path
    char_filename = f"{output_folder}/char_{char_index}.png"
    cv2.imwrite(char_filename, roi)

    print(f"Character '{char}' saved: {char_filename}")

    char_index += 1

# For closing the windows
cv2.waitKey(0)
cv2.destroyAllWindows()