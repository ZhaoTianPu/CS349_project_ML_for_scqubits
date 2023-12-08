import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 100)  # Assuming 2 input features
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 3)  # Assuming 3 output values

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Logistic activation
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)
# Load the saved model states
input_size = 2  # Number of features
hidden_size = 100  # Number of features in hidden state
num_layers = 2  # Number of stacked LSTM layers
output_size = 3  # Number of output values
lstm_model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)
mlp_model =  MLP()
lstm_model.load_state_dict(torch.load('lstm_model.pth'))
mlp_model.load_state_dict(torch.load('model.pth'))

lstm_model.eval()
mlp_model.eval()


# Load the test data
test_df = pd.read_csv('test_data.csv')
X_test = test_df.iloc[:, :-3].values
y_test = test_df.iloc[:, -3:].values
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    # Reshape for LSTM if needed
    lstm_predictions = lstm_model(X_test_tensor.unsqueeze(1)).numpy()
    mlp_predictions = mlp_model(X_test_tensor).numpy()
# Calculate relative errors
relative_errors_lstm = np.abs(lstm_predictions - y_test) / (np.abs(y_test) + 1e-10)
relative_errors_mlp = np.abs(mlp_predictions - y_test) / (np.abs(y_test) + 1e-10)
average_relative_error_mlp = np.mean(relative_errors_mlp)
average_relative_error_lstm = np.mean(relative_errors_lstm)

# Print the average relative errors
print(f"Average Relative Error (MLP): {average_relative_error_mlp}")
print(f"Average Relative Error (LSTM): {average_relative_error_lstm}")
# Calculate error ratios
error_ratios = relative_errors_lstm / (relative_errors_mlp + 1e-10)
print(error_ratios )
# Assuming E_j is the first column and E_c is the second column in X_test
E_c = X_test[:, 0]
E_j = X_test[:, 1]

# Assuming error_ratios is a Nx3 array where N is the number of test samples
# and each column corresponds to the error ratio for each of the 3 eigenvalues
color_below_one = 'blue'
color_above_one = 'red'
# Plotting
for i in range(3):  # Assuming 3 eigenvalues
    plt.figure(figsize=(10, 6))
    colors = [color_above_one if ratio >= 1 else color_below_one for ratio in error_ratios[:, i]]
    plt.scatter(E_j, E_c, color=colors)
    plt.xlabel('E_j')
    plt.ylabel('E_c')
    plt.title(f'Error Ratio by E_j and E_c for Eigenvalue {i+1}')
    plt.show()

# for i in range(3):  # Assuming 3 eigenvalues
#     plt.figure(figsize=(10, 6))
#     sc = plt.scatter(E_j, E_c, c=error_ratios[:, i],
#                      norm=colors.LogNorm(), cmap='viridis')
#     plt.colorbar(sc, label=f'Error Ratio for Eigenvalue {i+1} (LSTM/MLP, log scale)')
#     plt.xlabel('E_j')
#     plt.ylabel('E_c')
#     plt.title(f'Error Ratio by E_j and E_c for Eigenvalue {i+1}')
#     plt.show()