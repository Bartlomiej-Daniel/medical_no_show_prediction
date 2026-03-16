from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Load data
data_dir = Path(__file__).parent.parent
data_path = data_dir / "data" / "processed"

x_train = pd.read_csv(data_path / "x_train.csv")
y_train = pd.read_csv(data_path / "y_train.csv")

x_test = pd.read_csv(data_path / "x_test.csv")
y_test = pd.read_csv(data_path / "y_test.csv")

# Scale features
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Neural network model
class no_show_nn(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        return self.network(x)


input_size = x_train.shape[1]

model = no_show_nn(input_size)

# Training setup
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

# Training loop
for epoch in range(epochs):
    outputs = model(x_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epochs % 10 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    predictions = model(x_test)

    predicted = (predictions > 0.5).float()

accurancy = (predicted == y_test).float().mean()

print("Test accurancy:", accurancy.item())