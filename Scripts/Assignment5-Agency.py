import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

# ====== Hyperparameters ======
EPOCHS = 100
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 4.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.65
LR = 0.001

# ====== Data Preparation ======
df = pd.read_csv("../final_datasets/pixel_counts_with_cv_prediction_big_5.csv")
df = df.drop(columns=["Image","Ocp_Letter"])

le_letters = LabelEncoder()
df["Expected_Letter"] = le_letters.fit_transform(df["Expected_Letter"])

le_rf = LabelEncoder()
df["CV_Prediction"] = le_rf.fit_transform(df["CV_Prediction"])

X = df.drop(columns=["Expected_Letter"]).values.astype(np.float32)
y = df["Expected_Letter"].values.astype(int)
rf_out = df[("CV_Prediction")].values.astype(int)

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

# ====== Initialize Model, Target, Optimizer ======
model = ImprovedDQN(X.shape[1], n_classes)
target_model = ImprovedDQN(X.shape[1], n_classes)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
epsilon = EPSILON_START

# ====== History for Visualization ======
history = {
    'epoch': [],
    'loss': [],
    'reward': [],
    'epsilon': []
}

# ====== Training Loop ======
for epoch in range(EPOCHS):
    indices = torch.randperm(X.size(0))
    total_loss = 0
    total_reward = 0

    for i in range(0, X.size(0), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        states = X[batch_idx]
        labels = y[batch_idx]
        rf_preds = rf_out[batch_idx]

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

# ====== Evaluation ======
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for i in range(X.size(0)):
        q_vals = model(X[i])
        pred = torch.argmax(q_vals).item()
        preds.append(pred)
        true_labels.append(y[i].item())

acc = accuracy_score(true_labels, preds)
print(f"\nFinal Accuracy of Improved DQN Agent: {acc:.4f}")

# ====== Letter-wise Accuracy ======
letter_acc = {}
true_labels = np.array(true_labels)
preds = np.array(preds)

for label in np.unique(true_labels):
    mask = true_labels == label
    correct = (preds[mask] == true_labels[mask]).sum()
    letter_acc[le_letters.inverse_transform([label])[0]] = correct / mask.sum()

print("\nLetter-wise Accuracies:")
for letter, acc in sorted(letter_acc.items()):
    print(f"{letter}: {acc:.4f}")

# ====== Training Graphs ======
plt.figure(figsize=(18, 5))

# Loss Graph
plt.subplot(1, 3, 1)
plt.plot(history['epoch'], history['loss'], label='Loss', color='blue')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Reward Graph
plt.subplot(1, 3, 2)
plt.plot(history['epoch'], history['reward'], label='Reward', color='green')
plt.title("Total Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)

# Epsilon Graph
plt.subplot(1, 3, 3)
plt.plot(history['epoch'], history['epsilon'], label='Epsilon', color='red')
plt.title("Epsilon Decay")
plt.xlabel("Epoch")
plt.ylabel("Epsilon")
plt.grid(True)

plt.tight_layout()
plt.show()
