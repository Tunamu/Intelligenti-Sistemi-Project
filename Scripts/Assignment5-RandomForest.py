import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

file = pd.read_csv("../little_pixel_counts_new/little_pixel_counts_new_5.csv")
X = file.drop(columns=["Image","Expected_Letter", "Ocp_Letter"]).values
y = file["Expected_Letter"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)


#For Using Random Forest Method
rf_model = RandomForestClassifier()
rf_model.fit(X, y_encoded)
rf_pred = rf_model.predict(X)
rf_pred_transformed = le.inverse_transform(rf_pred)

file["Letter_From_Random_Forest"] = rf_pred_transformed
file.to_csv("../final_datasets/final_dataset_with_random_forest_little_5.csv", index=False)
print("Script Completed")