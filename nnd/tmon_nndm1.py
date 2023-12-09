

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import numpy as np

# Read the CSV files
train_df = pd.read_csv('train_data.csv')
valid_df = pd.read_csv('valid_data.csv')
test_df = pd.read_csv('test_data.csv')
# Assuming the last 3 columns are the y_data (eigenvalues)
# and the first 2 columns are the X_data (features)

# Extract X_train, y_train
X_train = train_df.iloc[:, :-3].values
y_train = train_df.iloc[:, -3:].values

# Extract X_valid, y_valid
X_valid = valid_df.iloc[:, :-3].values
y_valid = valid_df.iloc[:, -3:].values

# Extract X_test, y_test
X_test = test_df.iloc[:, :-3].values
y_test = test_df.iloc[:, -3:].values

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32  # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Assuming 2 input features
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)  # Assuming 3 output values

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Logistic activation
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
# Training Loop
num_epochs = 3000  # You can adjust this
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update the learning rate
    # Validation
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for X_batch, y_batch in valid_loader:
            y_pred = model(X_batch)
            valid_loss += criterion(y_pred, y_batch).item()
        valid_loss /= len(valid_loader)

    # Early Stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = model.state_dict()  # Save the best model

    # Optional: Print Epoch and Loss Information
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss:.4f}')

# Load the best model
model.load_state_dict(best_model)

model.eval()
torch.save(model.state_dict(), 'model.pth')

with torch.no_grad():
    y_pred = []
    for X_batch, _ in test_loader:
        y_batch_pred = model(X_batch)
        y_pred.extend(y_batch_pred.numpy())
    y_pred = np.array(y_pred)

# Calculate the relative errors
relative_errors = np.abs(y_pred - y_test) / np.abs(y_test)

# Flatten the arrays to ensure all individual errors are accounted for
relative_errors_flat = relative_errors.flatten()

# Calculate mean squared error of the relative errors
mse_relative = np.mean(np.square(relative_errors_flat))

print("Mean Squared Error of Relative Errors:", mse_relative)