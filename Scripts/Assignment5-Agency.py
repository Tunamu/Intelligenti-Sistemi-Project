import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# ====== Hyperparameters ======
EPOCHS = 100
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 4.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.65
LR = 0.001
N_SPLITS = 10

# ====== Data Preparation ======
df = pd.read_csv("../final_datasets/pixel_counts_with_cv_prediction_big_5.csv")
df = df.drop(columns=["Image", "Ocp_Letter"])

le_letters = LabelEncoder()
df["Expected_Letter"] = le_letters.fit_transform(df["Expected_Letter"])

le_rf = LabelEncoder()
df["CV_Prediction"] = le_rf.fit_transform(df["CV_Prediction"])

X = df.drop(columns=["Expected_Letter"]).values.astype(np.float32)
y = df["Expected_Letter"].values.astype(int)
rf_out = df["CV_Prediction"].values.astype(int)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X)
y = torch.tensor(y)
rf_out = torch.tensor(rf_out)

n_classes = len(np.unique(y))

# ====== DQN Model Definition ======
class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ====== Cross-Validation Setup ======
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_accuracies = []
all_histories = []
letter_accuracies_per_fold = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n==== Fold {fold+1} ====")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf_train, rf_test = rf_out[train_index], rf_out[test_index]

    model = ImprovedDQN(X.shape[1], n_classes)
    target_model = ImprovedDQN(X.shape[1], n_classes)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    epsilon = EPSILON_START

    history = {'epoch': [], 'loss': [], 'reward': [], 'epsilon': []}

    for epoch in range(EPOCHS):
        indices = torch.randperm(X_train.size(0))
        total_loss = 0
        total_reward = 0

        for i in range(0, X_train.size(0), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            states = X_train[batch_idx]
            labels = y_train[batch_idx]
            rf_preds = rf_train[batch_idx]

            q_values = model(states)
            with torch.no_grad():
                next_q_values = target_model(states)

            targets = q_values.clone().detach()

            for j in range(states.size(0)):
                if random.random() < epsilon:
                    action = random.randint(0, n_classes - 1)
                else:
                    action = torch.argmax(q_values[j]).item()

                reward = 8 if action == labels[j].item() else (1 if action == rf_preds[j].item() else -5)
                total_reward += reward

                max_next_q = torch.max(next_q_values[j]).item()
                targets[j, action] = reward + GAMMA * max_next_q

            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Total Reward: {total_reward}")

        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss)
        history['reward'].append(total_reward)
        history['epsilon'].append(epsilon)

    all_histories.append(history)

    # ====== Evaluation ======
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for i in range(X_test.size(0)):
            q_vals = model(X_test[i])
            pred = torch.argmax(q_vals).item()
            preds.append(pred)
            true_labels.append(y_test[i].item())

    acc = accuracy_score(true_labels, preds)
    fold_accuracies.append(acc)
    print(f"\nFold {fold+1} Accuracy: {acc:.4f}")

    # ====== Letter-wise Accuracy ======
    letter_acc = {}
    true_labels_np = np.array(true_labels)
    preds_np = np.array(preds)

    for label in np.unique(true_labels_np):
        mask = true_labels_np == label
        correct = (preds_np[mask] == true_labels_np[mask]).sum()
        letter = le_letters.inverse_transform([label])[0]
        acc_letter = correct / mask.sum()
        letter_acc[letter] = acc_letter

    letter_accuracies_per_fold.append(letter_acc)

    print("\nLetter-wise Accuracies:")
    for letter, acc_l in sorted(letter_acc.items()):
        print(f"{letter}: {acc_l:.4f}")

# ====== Visualization ======
plt.figure(figsize=(18, 5 * N_SPLITS))

for i, hist in enumerate(all_histories):
    plt.subplot(N_SPLITS, 3, i * 3 + 1)
    plt.plot(hist['epoch'], hist['loss'], color='blue')
    plt.title(f"Fold {i+1} - Loss")
    plt.grid(True)

    plt.subplot(N_SPLITS, 3, i * 3 + 2)
    plt.plot(hist['epoch'], hist['reward'], color='green')
    plt.title(f"Fold {i+1} - Reward")
    plt.grid(True)

    plt.subplot(N_SPLITS, 3, i * 3 + 3)
    plt.plot(hist['epoch'], hist['epsilon'], color='red')
    plt.title(f"Fold {i+1} - Epsilon")
    plt.grid(True)

plt.tight_layout()
plt.show()

# ====== Fold Accuracy Summary ======
print("\n==== Cross-Validation Summary ====")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1}: {acc:.4f}")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")

# Optional: Letter-wise accuracy summary across folds
print("\n==== Letter-wise Accuracies Per Fold ====")
for i, acc_dict in enumerate(letter_accuracies_per_fold):
    print(f"\nFold {i+1}")
    for letter, acc in sorted(acc_dict.items()):
        print(f"{letter}: {acc:.4f}")

# ====== Harf Bazında Ortalama Doğruluk Grafiği ======
# Fold'lardaki letter accuracy'leri dataframe'e çevir
letter_df = pd.DataFrame(letter_accuracies_per_fold)
avg_letter_acc = letter_df.mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_letter_acc.index, y=avg_letter_acc.values, palette="viridis")
plt.title("Average Letter-wise Accuracy Over Folds")
plt.xlabel("Letter")
plt.ylabel("Average Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# ====== Heatmap: Fold x Harf Accuracy ======
plt.figure(figsize=(14, 8))
sns.heatmap(letter_df.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Letter-wise Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Letter")
plt.yticks(rotation=0)
plt.xticks(ticks=np.arange(N_SPLITS)+0.5, labels=[f"Fold {i+1}" for i in range(N_SPLITS)], rotation=45)
plt.tight_layout()
plt.show()
