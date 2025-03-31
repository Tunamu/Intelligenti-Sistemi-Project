import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 📌 1️⃣ CSV Dosyanı Oku
csv_path = ("big_pixel_counts_new_5.csv")
df = pd.read_csv(csv_path)

# 📌 2️⃣ Giriş (X) ve Çıkış (y) Sütunlarını Ayır
X = df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter"]).values  # Görsel adı ve metin sütunları hariç
y = df["Expected_Letter"].values  # Etiket olarak beklenen harf

# 📌 3️⃣ Etiketleri (Harfleri) Sayıya Çevir
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 📌 4️⃣ Veriyi Normalize Et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 📌 5️⃣ Eğitim ve Test Verisine Böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 6️⃣ PyTorch Dataset Sınıfını Tanımla
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 📌 7️⃣ DataLoader Oluştur
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 📌 8️⃣ Model Listesi
input_size = X.shape[1]  # Özellik sayısı
num_classes = len(label_encoder.classes_)  # Sınıf sayısı

models = {
    "MLP": nn.Sequential(
        nn.Linear(input_size, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes)
    ),
    "CNN": nn.Sequential(
        nn.Linear(input_size, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )
}

# 📌 9️⃣ Eğitim Fonksiyonu
def train_model(model, train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

# 📌 🔟 Accuracy Hesaplama
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, axis=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return round((correct / total) * 100, 2)

# 📌 1️⃣1️⃣ Modelleri Eğit ve Test Et
results = []
for model_name, model in models.items():
    print(f"\n🚀 {model_name} Eğitiliyor...")
    train_model(model, train_loader)
    acc = evaluate_model(model, test_loader)
    results.append([model_name, acc])

# 📌 1️⃣2️⃣ Sonuçları Göster
print("\n📊 PyTorch Modellerinin Test Doğrulukları:")
print(results)
