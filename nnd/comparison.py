import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import scqubits as scq
import numpy as np
import time
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

def nnd():
    mlp_model =  MLP()
    mlp_model.load_state_dict(torch.load('model.pth'))
    mlp_model.eval()


    # Load the test data
    test_df = pd.read_csv('test_data.csv')
    X_test = test_df.iloc[:, :-3].values
    y_test = test_df.iloc[:, -3:].values
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        # Reshape for LSTM if needed
        mlp_predictions = mlp_model(X_test_tensor).numpy()
    # Calculate relative errors
    relative_errors_mlp = np.abs(mlp_predictions - y_test) / (np.abs(y_test) + 1e-10)
    average_relative_error_mlp = np.mean(relative_errors_mlp)

    # Print the average relative errors
    print(f"Average Relative Error (MLP): {average_relative_error_mlp}")
    return relative_errors_mlp
def exact_dia():
    test_df = pd.read_csv('test_data.csv')
    X_test = test_df.iloc[:, :-3].values
    for data_point in X_test:
        E_c = np.float64(data_point[1])  # Charging energy
        E_j = np.float64(data_point[0])  # Josephson energy

        # Create a Transmon qubit instance
        transmon = scq.Transmon(
            EJ=E_j,
            EC=E_c,
            ng=0,     # offset charge, set to 0 for simplicity
            ncut=150  # number of charge states included in the basis, may need adjustment
        )

        # Calculate the first n eigenvalues
        eigenvalues = transmon.eigenvals(evals_count=4)
        eigenvalues = np.diff(eigenvalues)

num_iterations = 100
total_runtime_nnd = 0
total_runtime_exact_dia = 0
# Measure the runtime of nnd over 100 iterations
for _ in range(num_iterations):
    start_time = time.time()
    relative_error = nnd()
    print(relative_error)
    total_runtime_nnd += time.time() - start_time

# Measure the runtime of exact_dia over 100 iterations
for _ in range(num_iterations):
    start_time = time.time()
    exact_dia()
    total_runtime_exact_dia += time.time() - start_time

# Compute the average runtime
average_runtime_nnd = total_runtime_nnd / num_iterations
average_runtime_exact_dia = total_runtime_exact_dia / num_iterations
print(f"Average Runtime of nnd over {num_iterations} iterations: {average_runtime_nnd} seconds")
print(f"Average Runtime of exact_dia over {num_iterations} iterations: {average_runtime_exact_dia} seconds")
