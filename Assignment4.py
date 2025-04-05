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

# CSV Dosyasını Oku
file_path = "little_pixel_counts_new_10.csv"
df = pd.read_csv(file_path)

# Özellikleri (X) ve Etiketleri (y) Ayır
X = df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter"]).values
y = df["Expected_Letter"].values

# Etiketleri Sayıya Çevir
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize Et
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 10-Fold Ayarla
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

# Scikit-learn Modelleri
sklearn_models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Network": MLPClassifier(max_iter=3000)
}

# PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PyTorch Modelleri
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

# PyTorch Eğitim
def train_model(model, train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

# PyTorch Değerlendirme
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

    return round((correct / total) * 100, 4), np.array(y_true), np.array(y_pred)

# Model Eğitim/Test
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    fold_results = []

    for model_name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_per_letter = {}
        for letter in np.unique(y_test):
            mask = y_test == letter
            acc = accuracy_score(y_test[mask], y_pred[mask])
            accuracy_per_letter[label_encoder.inverse_transform([letter])[0]] = round(acc, 4)

        entry = {"Fold": fold + 1, "Model": model_name}
        entry.update(accuracy_per_letter)
        fold_results.append(entry)

    train_loader = DataLoader(CustomDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(CustomDataset(X_test, y_test), batch_size=32, shuffle=False)

    for model_name, model in pytorch_models.items():
        print(f"\n🚀 {model_name} - Fold {fold+1}")
        train_model(model, train_loader)
        acc, y_true, y_pred = evaluate_model(model, test_loader)

        accuracy_per_letter = {}
        for letter in np.unique(y_true):
            mask = y_true == letter
            acc = accuracy_score(y_true[mask], y_pred[mask])
            accuracy_per_letter[label_encoder.inverse_transform([letter])[0]] = round(acc, 4)

        entry = {"Fold": fold + 1, "Model": model_name}
        entry.update(accuracy_per_letter)
        fold_results.append(entry)

    fold_df = pd.DataFrame(fold_results)
    letter_cols = [col for col in fold_df.columns if col not in ["Fold", "Model"]]
    fold_avg_row = {"Fold": fold + 1, "Model": f"Fold_{fold+1}_Avg"}
    for col in letter_cols:
        fold_avg_row[col] = round(fold_df[col].mean(), 4)
    fold_avg_row["Fold_Model_Avg"] = round(fold_df[letter_cols].mean(axis=1).mean(), 4)

    results.extend(fold_results)
    results.append(fold_avg_row)

# Sonuçları Derle
results_df = pd.DataFrame(results)
letter_columns = [col for col in results_df.columns if col not in ["Fold", "Model", "Fold_Model_Avg"]]

if "Fold_Model_Avg" not in results_df.columns:
    results_df["Fold_Model_Avg"] = results_df[letter_columns].mean(axis=1)
else:
    results_df["Fold_Model_Avg"] = results_df["Fold_Model_Avg"].fillna(results_df[letter_columns].mean(axis=1))

# Model ortalamaları (her harf için + genel)
model_avg_df = results_df[results_df["Model"].isin(list(sklearn_models.keys()) + list(pytorch_models.keys()))]
model_avg_df = model_avg_df.groupby("Model")[letter_columns + ["Fold_Model_Avg"]].mean().round(4).reset_index()
model_avg_df.insert(0, "Fold", "")
model_avg_df["Model"] = model_avg_df["Model"] + "_Avg"

# Harf ortalaması
letter_avg_df = pd.DataFrame(results_df[letter_columns].mean()).T.round(4)
letter_avg_df.insert(0, "Model", "Letter_Avg")
letter_avg_df.insert(0, "Fold", "")

# Genel ortalama
overall_avg = results_df[results_df["Model"] != "Overall_Avg"]["Fold_Model_Avg"].mean()
overall_avg_df = pd.DataFrame([{
    "Fold": "",
    "Model": "Overall_Avg",
    **{col: "" for col in letter_columns},
    "Fold_Model_Avg": round(overall_avg, 4)
}])

# Hepsini birleştir ve CSV'ye yaz
final_df = pd.concat([results_df, model_avg_df, letter_avg_df, overall_avg_df], ignore_index=True)
final_df.to_csv("simplified_model_results_new_10.csv", index=False, float_format="%.4f")

print("✅ Tüm sonuçlar 4 ondalıklı olarak ve model ortalamalarıyla kaydedildi!")
