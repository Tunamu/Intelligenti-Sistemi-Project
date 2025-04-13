import pandas as pd
import numpy as np

file = pd.read_csv("../final_datasets/pixel_counts_with_cv_prediction_little_5.csv")
equal = 0
not_equal = 0

for i in file.values:
    if i[1] == i[-1]:
        equal += 1
    else:
        not_equal += 1

print(equal, not_equal)