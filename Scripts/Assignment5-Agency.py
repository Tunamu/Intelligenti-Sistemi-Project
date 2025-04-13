import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random

# 1. Veri okuma ve hazırlık
df = pd.read_csv("../final_datasets/final_dataset_with_random_forest_big_5.csv")
X = df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter","Letter_From_Random_Forest"]).values
y = LabelEncoder().fit_transform(df["Expected_Letter"].values)

# 2. Aksiyonlar (parametre kombinasyonları)
param_space = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, None]
}
actions = [(n, d) for n in param_space["n_estimators"] for d in param_space["max_depth"]]
n_actions = len(actions)

# 3. Q-learning parametreleri
q_table = np.zeros(n_actions)
alpha = 0.1
gamma = 0.9
epsilon = 0.2
n_episodes = 50

# Kayıt için
acc_per_episode = []
best_acc = 0
best_action = None

# 4. Q-learning döngüsü
for episode in range(n_episodes):
    if random.uniform(0, 1) < epsilon:
        action_idx = random.randint(0, n_actions - 1)
    else:
        action_idx = np.argmax(q_table)

    n_est, depth = actions[action_idx]

    # 10-fold cross-validation
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc = np.mean(scores)

    # Q-table güncelleme
    reward = acc
    old_value = q_table[action_idx]
    next_max = np.max(q_table)
    q_table[action_idx] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

    acc_per_episode.append(acc)

    # En iyi sonucu güncelle
    if acc > best_acc:
        best_acc = acc
        best_action = (n_est, depth)

    print(f"Episode {episode+1}: Accuracy = {acc:.4f}, Params = (n_estimators={n_est}, max_depth={depth})")

# 5. Sonuç
print("\nBest Params (Q-Learning):", best_action)
print("Best 10-Fold CV Accuracy:", best_acc)

# 6. Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_episodes+1), acc_per_episode, marker='o')
plt.title("Accuracy per Episode (10-Fold Cross-Validation)")
plt.xlabel("Episode")
plt.ylabel("Average Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_per_episode.png")
plt.show()
