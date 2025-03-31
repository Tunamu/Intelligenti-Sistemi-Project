import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# CSV dosyasını oku
file_path = "big_pixel_counts_new_5.csv"  # Eğer dosya başka bir dizindeyse yolu değiştir
df = pd.read_csv(file_path)

# Gereksiz sütunları temizle
df_cleaned = df.drop(columns=["Image", "Ocp_Letter"])  # Görüntü ismi ve OCR sonucu gereksiz

# X (özellikler) ve y (etiketler) ayır
X = df_cleaned.drop(columns=["Expected_Letter"])  # Tüm özellik sütunları
y = df_cleaned["Expected_Letter"]  # Hedef değişken

# Harfleri sayısal değerlere çevir (Label Encoding)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Harfleri 0,1,2,... gibi sayılara çeviriyoruz

# Eğitim ve test setlerine ayır (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelleri eğitip test edelim
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Modeli eğit
    y_pred = model.predict(X_test)  # Test verisiyle tahmin yap
    acc = accuracy_score(y_test, y_pred)  # Doğruluk skorunu hesapla
    results[name] = acc
    print(f"{name}: {acc:.4f}")  # Sonuçları yazdır

# Sonuçları çıktı olarak göster
print("Model Performansları:", results)