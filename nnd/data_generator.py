from classify.tmon_classifier import generate_data
import scqubits as scq
import numpy as np
train_set, valid_set, test_set = generate_data()

def obtain_data(data_set):
    train_set_numerical = np.array(data_set[:, :-1], dtype=float)
    labels = data_set[:, -1]  # Keep labels separate

    # Extracting data for each class
    charge_data = train_set_numerical[labels == 'charge']
    transmon_data = train_set_numerical[labels == 'transmon']
    n_eigenvalues = 3  # for example, calculate the first 5 eigenvalues
    # Lists to store input features (E_c and E_j) and corresponding eigenvalues
    x_data = []
    y_data = []
    # Loop over the training set
    for data_point in transmon_data:
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
        eigenvalues = transmon.eigenvals(evals_count=n_eigenvalues)

        # Append the input features and corresponding eigenvalues to the lists
        x_data.append([E_c, E_j])
        y_data.append(eigenvalues)

    # Convert lists to numpy arrays for easier handling in ML models
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    # x_data and y_data are now ready to be used for training a neural network
    return x_data,y_data
X_train,  y_train = obtain_data(train_set)
X_valid,  y_valid = obtain_data(valid_set)
X_test,  y_test = obtain_data(test_set)

import pandas as pd

# Convert the numpy arrays to pandas DataFrames
train_df = pd.DataFrame(np.hstack((X_train, y_train)))
valid_df = pd.DataFrame(np.hstack((X_valid, y_valid)))
test_df = pd.DataFrame(np.hstack((X_test, y_test)))

# Saving the dataframes to CSV files
train_df.to_csv('train_data.csv', index=False)
valid_df.to_csv('valid_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)