import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

# 1. Veriyi oku
df = pd.read_csv('../big_pixel_counts_new/big_pixel_counts_new_5.csv')

# 2. Özellikleri ve hedefi ayır
X = df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter"])
y = df["Expected_Letter"]

# 3. K-fold setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 4. Boş tahmin dizisi
predictions = np.empty(len(df), dtype=object)

# 5. Her fold için eğitim ve test
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train = y.iloc[train_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[test_index] = y_pred

# 6. Tahminleri orijinal dataframe'e ekle
df['CV_Prediction'] = predictions

# 7. Yeni dosyayı kaydet
df.to_csv('../final_datasets/pixel_counts_with_cv_prediction_big_5.csv', index=False)
