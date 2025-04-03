import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 📌 1️⃣ CSV Dosyasını Oku
file_path = "big_pixel_counts_new_2.csv"
df = pd.read_csv(file_path)

# 📌 2️⃣ Özellikleri (X) ve Etiketleri (y) Ayır
X = df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter"]).values
y = df["Expected_Letter"].values

# 📌 3️⃣ Etiketleri (Harfleri) Sayıya Çevir
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 📌 4️⃣ Veriyi Normalize Et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 📌 5️⃣ 10-Fold Çapraz Doğrulama Ayarla
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

# 📌 6️⃣ Scikit-Learn Modelleri
sklearn_models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Network": MLPClassifier(max_iter=3000)
}

# 📌 7️⃣ PyTorch Dataset Sınıfını Tanımla
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 📌 8️⃣ PyTorch Modelleri
input_size = X.shape[1]
num_classes = len(label_encoder.classes_)

pytorch_models = {
    "MLP (PyTorch)": nn.Sequential(
        nn.Linear(input_size, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, num_classes)
    ),
    "CNN (PyTorch)": nn.Sequential(
        nn.Linear(input_size, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )
}

# 📌 9️⃣ PyTorch Model Eğitimi Fonksiyonu
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

# 📌 🔟 PyTorch Model Değerlendirme Fonksiyonu
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, axis=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    return round((correct / total) * 100, 2), np.array(y_true), np.array(y_pred)

# 📌 1️⃣1️⃣ 10-Fold Çapraz Doğrulama (Tüm Modeller İçin)
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # 📌 1️⃣2️⃣ Scikit-Learn Modellerini Eğit ve Test Et
    for model_name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Harf bazlı doğruluk hesapla
        accuracy_per_letter = {}
        for letter in np.unique(y_test):
            letter_mask = y_test == letter
            letter_accuracy = accuracy_score(y_test[letter_mask], y_pred[letter_mask])
            accuracy_per_letter[label_encoder.inverse_transform([letter])[0]] = round(letter_accuracy, 2)

        # Sonuçları kaydet
        result_entry = {"Fold": fold + 1, "Model": model_name}
        result_entry.update(accuracy_per_letter)
        results.append(result_entry)

    # 📌 1️⃣3️⃣ PyTorch Modellerini Eğit ve Test Et
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for model_name, model in pytorch_models.items():
        print(f"\n🚀 {model_name} - Fold {fold+1} Eğitiliyor...")
        train_model(model, train_loader)
        acc, y_true, y_pred = evaluate_model(model, test_loader)

        # PyTorch için harf bazlı doğruluk hesaplama
        accuracy_per_letter = {}
        for letter in np.unique(y_true):
            letter_mask = y_true == letter
            letter_accuracy = accuracy_score(y_true[letter_mask], y_pred[letter_mask])
            accuracy_per_letter[label_encoder.inverse_transform([letter])[0]] = round(letter_accuracy, 2)

        # Sonuçları kaydet
        result_entry = {"Fold": fold + 1, "Model": model_name}
        result_entry.update(accuracy_per_letter)
        results.append(result_entry)

# 📌 1️⃣4️⃣ Sonuçları CSV'ye Kaydet
results_df = pd.DataFrame(results)
output_path = "combined_model_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\n📊 Sonuçlar kaydedildi: {output_path}")
