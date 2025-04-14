import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import legend
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# Parameters for Deep Q Learning
Branch_Size = 50
Fold_Size = 10
Learn_Rate = 0.003
First_Epsilon_Value= 2.5
Epsilon_Min_Value= 0.1
Epsilon_Mult_Factor = 0.70
Batch_Size = 128
Gamma = 0.90

df = pd.read_csv("../final_datasets/pixel_counts_with_cv_prediction_big_5.csv")
df = df.drop(columns=["Image", "Ocp_Letter"])

le_letters = LabelEncoder()
df["Expected_Letter"] = le_letters.fit_transform(df["Expected_Letter"]) # Converting Letters into numbers

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

class ImprovedDQN(nn.Module):
    def __init__(self, feature_size, letter_size):
        super(ImprovedDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, 512), # Input estimates as 512 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, letter_size) # Output size same as letter size
        )

    # How the system works when we give data
    def forward(self, x):
        return self.net(x)

skf = StratifiedKFold(n_splits=Fold_Size, shuffle=True, random_state=42) # For testing same packets "random_state=42"
fold_accuracies = [] # For holding accurate of every fold
all_histories = []  # Training process values
letter_accuracies_per_fold = [] # Letter Based Accuracies

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1} testing...")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf_train, rf_test = rf_out[train_index], rf_out[test_index]

    model = ImprovedDQN(X.shape[1], n_classes)
    target_model = ImprovedDQN(X.shape[1], n_classes)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=Learn_Rate)
    criterion = nn.MSELoss()
    epsilon = First_Epsilon_Value

    history = {'Branch_Size': [], 'loss': [], 'reward': [], 'epsilon': []}

    for Branch_Size in range(Branch_Size):
        indices = torch.randperm(X_train.size(0))
        total_loss = 0
        total_reward = 0

        for i in range(0, X_train.size(0), Batch_Size):
            batch_idx = indices[i:i+Batch_Size]
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

                reward = 8 if action == labels[j].item() else (4 if action == rf_preds[j].item() else -5)
                total_reward += reward

                max_next_q = torch.max(next_q_values[j]).item()
                targets[j, action] = reward + Gamma * max_next_q

            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if Branch_Size % 5 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(Epsilon_Min_Value, epsilon * Epsilon_Mult_Factor)

        # print(f"Branch_Size {Branch_Size+1}/{Branch_Size} - Loss: {total_loss:.4f} - Total Reward: {total_reward}")

        history['Branch_Size'].append(Branch_Size + 1)
        history['loss'].append(total_loss)
        history['reward'].append(total_reward)
        history['epsilon'].append(epsilon)

    all_histories.append(history)

    # For using test dataset we create some arrays
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
    print(f"Fold {fold+1} Accuracy: %{acc*100:.2f}\n")

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

    # print("\nLetter-wise Accuracies:")
    # for letter, acc_l in sorted(letter_acc.items()):
    #     print(f"{letter}: {acc_l:.4f}")

plt.figure(figsize=(18, 5 * Fold_Size))

for i, hist in enumerate(all_histories):
    plt.subplot(Fold_Size, 3, i * 3 + 1)
    plt.plot(hist['Branch_Size'], hist['loss'], color='blue')
    plt.title(f"Fold {i+1} - Loss")
    plt.grid(True)

    plt.subplot(Fold_Size, 3, i * 3 + 2)
    plt.plot(hist['Branch_Size'], hist['reward'], color='green')
    plt.title(f"Fold {i+1} - Reward")
    plt.grid(True)

    plt.subplot(Fold_Size, 3, i * 3 + 3)
    plt.plot(hist['Branch_Size'], hist['epsilon'], color='red')
    plt.title(f"Fold {i+1} - Epsilon")
    plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== Summary")
# for i, acc in enumerate(fold_accuracies):
#    print(f"Fold {i+1}: {acc:.4f}")
print(f"Average Accuracy: %{np.mean(fold_accuracies)*100:.2f}")

letter_df = pd.DataFrame(letter_accuracies_per_fold)
avg_letter_acc = letter_df.mean().sort_index()

print("\n==== Average Letter Accuracies")
for letter, acc in avg_letter_acc.items():
    print(f"{letter}: %{acc*100:.2f}")

# Optional: Letter-wise accuracy summary across folds
'''print("\n==== Letter Accuracies Per Fold")
for i, acc_dict in enumerate(letter_accuracies_per_fold):
    print(f"\nFold {i+1}")
    for letter, acc in sorted(acc_dict.items()):
        print(f"{letter}: {acc:.4f}")
'''

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_letter_acc.index, y=avg_letter_acc.values, hue=avg_letter_acc.index, palette="viridis", legend=False)
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
plt.xticks(ticks=np.arange(Fold_Size)+0.5, labels=[f"Fold {i+1}" for i in range(Fold_Size)], rotation=45)
plt.tight_layout()
plt.show()
