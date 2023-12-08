import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
# Function to generate data
def generate_data():
    # Constants
    pi = np.pi
    num_samples_per_class = 1500

    # Generate data for "transmon"
    E_j_transmon = np.random.uniform(80 * 2 * pi, 90 * 2 * pi, num_samples_per_class)
    E_c_transmon = np.random.uniform(0.1 * 2 * pi, 1 * 2 * pi, num_samples_per_class)
    transmon_data = np.column_stack((E_j_transmon, E_c_transmon))
    transmon_labels = np.array(["transmon"] * num_samples_per_class)

    # Generate data for "charge"
    E_j_charge = np.random.uniform(0.1 * 2 * pi, 1 * 2 * pi, num_samples_per_class)
    E_c_charge = np.random.uniform(0.1 * 2 * pi, 1 * 2 * pi, num_samples_per_class)
    charge_data = np.column_stack((E_j_charge, E_c_charge))
    charge_labels = np.array(["charge"] * num_samples_per_class)

    # Combine and shuffle the data
    data = np.vstack((transmon_data, charge_data))
    labels = np.concatenate((transmon_labels, charge_labels))
    combined = np.column_stack((data, labels))
    np.random.shuffle(combined)

    # Splitting the dataset into train, validation, and test sets
    train, test_valid = train_test_split(combined, test_size=0.4, random_state=42)
    valid, test = train_test_split(test_valid, test_size=0.5, random_state=42)

    return train, valid, test

# # Generate the data
# train_set, valid_set, test_set = generate_data()
#
# # Preprocess the data: Split attributes and labels, and encode labels
# def preprocess_data(dataset):
#     X = dataset[:, :-1].astype(float)  # Attributes
#     y = dataset[:, -1]  # Labels
#
#     # Convert labels to numeric values
#     encoder = LabelEncoder()
#     y_encoded = encoder.fit_transform(y)
#
#     return X, y_encoded
#
# X_train, y_train = preprocess_data(train_set)
# X_test, y_test = preprocess_data(test_set)
#
# # Create and train KNN model
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, y_train)
#
# # Make predictions and evaluate the model
# predictions = knn.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, predictions))
# print("Classification Report:")
# print(classification_report(y_test, predictions))
#
#
# train_set_numerical = np.array(train_set[:, :-1], dtype=float)
# labels = train_set[:, -1]  # Keep labels separate
#
# # Extracting data for each class
# charge_data = train_set_numerical[labels == 'charge']
# transmon_data = train_set_numerical[labels == 'transmon']
# # Plotting
# plt.figure(figsize=(10, 6))
#
# # Plot charge data in red
# # plt.scatter(charge_data[:, 0], charge_data[:, 1], color='red', label='Charge')
#
# # Plot transmon data in blue
# plt.scatter(transmon_data[:, 0], transmon_data[:, 1], color='blue', label='Transmon')
#
# # Adding labels and title
# plt.xlabel('E_j')
# plt.ylabel('E_c')
# plt.title('Training Data Visualization')
# plt.legend()
#
# # Show the plot
# plt.show()