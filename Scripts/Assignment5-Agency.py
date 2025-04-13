import numpy as np
import pandas as pd
import string
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# --- 1. Ortam (Environment) ---
class LetterEnv:
    def __init__(self, data):
        self.df = data
        self.features = self.df.drop(columns=["Image", "Expected_Letter", "Ocp_Letter", "CV_Prediction"]).values
        self.labels = self.df["Expected_Letter"].values
        self.letters = list(string.ascii_uppercase)

        self.action_space = len(self.letters)
        self.state_space = self.features.shape[1]
        self.index = 0

    def reset(self):
        self.index = 0
        return self.features[self.index]

    def step(self, action):
        predicted_letter = self.letters[action]
        correct_letter = self.labels[self.index]

        reward = 1 if predicted_letter == correct_letter else -1

        self.index += 1
        done = self.index >= len(self.features)
        next_state = None if done else self.features[self.index]

        return next_state, reward, done


# --- 2. Sinir Ağı (Q-Network) ---
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Daha büyük bir ağ yapısı
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# --- 3. Ajan (DQN Agent) ---
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=10000)  # Daha büyük bir hafıza
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.BoolTensor(dones)

        # next_states içinde None varsa sıfırla
        next_states_tensor = torch.zeros((self.batch_size, self.state_dim))
        mask = torch.ones(self.batch_size, dtype=bool)
        for i, ns in enumerate(next_states):
            if ns is not None:
                next_states_tensor[i] = torch.FloatTensor(ns)
            else:
                mask[i] = False

        q_values = self.model(states).gather(1, actions)

        next_q_values = torch.zeros(self.batch_size, 1)
        if mask.sum() > 0:
            with torch.no_grad():
                next_q = self.model(next_states_tensor)
                next_q_values[mask] = next_q.max(dim=1)[0].unsqueeze(1)[mask]

        target_q_values = rewards + (self.gamma * next_q_values)
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- 4. Eğitim Döngüsü ---
def train_dqn(data_path, k=10, episodes=50):  # Epizod sayısını artırdık
    df = pd.read_csv(data_path)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Eğitimdeki toplam ödülleri depolamak için bir liste
    all_rewards = []

    # K-fold işlemi
    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        print(f"Fold {fold + 1}/{k}")
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        env_train = LetterEnv(train_data)
        env_test = LetterEnv(test_data)

        agent = DQNAgent(state_dim=env_train.state_space, action_dim=env_train.action_space)

        # Eğitim kısmı
        total_rewards = []  # Bu fold için ödüller
        for episode in range(episodes):
            state = env_train.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env_train.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")

        all_rewards.append(total_rewards)

        # Test kısmı
        correct_predictions = {letter: 0 for letter in string.ascii_uppercase}
        total_predictions = {letter: 0 for letter in string.ascii_uppercase}

        state = env_test.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env_test.step(action)
            predicted_letter = string.ascii_uppercase[action]
            correct_letter = env_test.labels[env_test.index - 1]

            if predicted_letter == correct_letter:
                correct_predictions[predicted_letter] += 1
            total_predictions[predicted_letter] += 1

            state = next_state

        # Her bir harf için doğruluk oranını yazdır
        print("Test Sonuçları (Harf Başına Doğruluk Yüzdesi):")
        for letter in string.ascii_uppercase:
            if total_predictions[letter] > 0:
                accuracy = correct_predictions[letter] / total_predictions[letter] * 100
                print(f"{letter}: {accuracy:.2f}%")

    # Grafikleştirme: Eğitimdeki toplam ödüller
    plt.figure(figsize=(10, 6))
    for fold_rewards in all_rewards:
        plt.plot(fold_rewards, label="Episod Ödülü")
    plt.title("DQN Eğitim Sonuçları (Toplam Ödüller)")
    plt.xlabel("Epizod")
    plt.ylabel("Toplam Ödül")
    plt.legend()
    plt.show()


# --- 5. Scripti çalıştır ---
if __name__ == "__main__":
    train_dqn("../final_datasets/pixel_counts_with_cv_prediction_big_5.csv", k=10, episodes=10)  # Epizod sayısını artırdık
