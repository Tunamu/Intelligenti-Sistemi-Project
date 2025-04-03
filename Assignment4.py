import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# CSV dosyasını oku
file_path = "big_pixel_counts_new_2.csv"
df = pd.read_csv(file_path)

# Gereksiz sütunları temizle
df_cleaned = df.drop(columns=["Image", "Ocp_Letter"])  # Görüntü ismi ve OCR sonucu gereksiz

# X (özellikler) ve y (etiketler) ayır
X = df_cleaned.drop(columns=["Expected_Letter"])
y = df_cleaned["Expected_Letter"]

# Harfleri sayısal değerlere çevir (Label Encoding)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Kullanılacak modeller
models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP Neural Network": MLPClassifier(max_iter=3000)
}

# 10-fold çapraz doğrulama
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Harf bazlı doğruluk hesapla
        accuracy_per_letter = {}
        for letter in np.unique(y_test):
            letter_mask = y_test == letter
            letter_accuracy = accuracy_score(y_test[letter_mask], y_pred[letter_mask])
            accuracy_per_letter[label_encoder.inverse_transform([letter])[0]] = round(letter_accuracy, 2)

        # Sonuçları sakla
        result_entry = {"Fold": fold + 1, "Model": model_name}
        result_entry.update(accuracy_per_letter)
        results.append(result_entry)

# Sonuçları CSV'ye kaydet
results_df = pd.DataFrame(results)
output_path = "letter_accuracy_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Sonuçlar kaydedildi: {output_path}")
