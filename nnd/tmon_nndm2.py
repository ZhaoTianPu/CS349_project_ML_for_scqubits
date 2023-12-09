import scqubits as scq
from classify.tmon_classifier import generate_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Assuming x is of shape (batch, seq, feature)
        output, (hidden, _) = self.lstm(x)
        output = self.fc(output[:, -1, :])  # Get the last time step
        return output

# Define the model
input_size = 2  # Number of features
hidden_size = 100  # Number of features in hidden state
num_layers = 2  # Number of stacked LSTM layers
output_size = 3  # Number of output values

lstm_model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)


def train_lstm_model(model, train_loader, valid_loader, num_epochs=2000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_valid_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            # Reshape input for LSTM
            X_batch = X_batch.unsqueeze(1)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                # Reshape input for LSTM
                X_batch = X_batch.unsqueeze(1)

                y_pred = model(X_batch)
                valid_loss += criterion(y_pred, y_batch).item()
            valid_loss /= len(valid_loader)

        # Early Stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()  # Save the best model

        # Print Epoch and Loss Information
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {valid_loss:.4f}')

    # Load the best model
    model.load_state_dict(best_model)
    return model
lstm_model = train_lstm_model(lstm_model, train_loader, valid_loader)

torch.save(lstm_model.state_dict(), 'lstm_model.pth')

lstm_model.eval()
with torch.no_grad():
    y_pred_lstm = []
    for X_batch, _ in test_loader:
        # Reshape for LSTM
        X_batch = X_batch.unsqueeze(1)
        y_batch_pred = lstm_model(X_batch)
        y_pred_lstm.extend(y_batch_pred.numpy())
    y_pred_lstm = np.array(y_pred_lstm)

# Calculate the relative errors for LSTM
relative_errors_lstm = np.abs(y_pred_lstm - y_test) / np.abs(y_test)

# Flatten the arrays to ensure all individual errors are accounted for
relative_errors_flat_lstm = relative_errors_lstm.flatten()

# Calculate mean squared error of the relative errors for LSTM
mse_relative_lstm = np.mean(np.square(relative_errors_flat_lstm))

print("LSTM Model - Mean Squared Error of Relative Errors:", mse_relative_lstm)

