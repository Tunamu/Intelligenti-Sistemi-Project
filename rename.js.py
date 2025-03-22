import os

# Klasör yolu
image_folder = "LittleCharacterRepository"

# Klasördeki tüm dosyaları al
for image_name in os.listdir(image_folder):
    old_path = os.path.join(image_folder, image_name)

    # Eğer dosya zaten "Big" ile başlıyorsa atla
    if image_name.startswith("little_"):
        continue

    # Yeni dosya adını oluştur
    new_name = "little_" + image_name
    new_path = os.path.join(image_folder, new_name)

    # Dosya adını değiştir
    os.rename(old_path, new_path)

print("Tüm dosya adları güncellendi!")