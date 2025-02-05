# Bruno Iochins Grisci
# February 2nd, 2025

import numpy as np
import pandas as pd

# Set the parameters for the dataset
n_samples = 500  # Number of samples
n_features = 150  # Total number of features
n_informative = 15  # Number of informative (relevant) features (5 per class)
n_classes = 3  # Number of distinct labels

# Generate random data for the informative features
np.random.seed(42)  # For reproducibility

# Initialize an empty feature matrix
X = np.zeros((n_samples, n_features))

# Split the dataset into three parts for the three classes
split_indices = np.split(np.arange(n_samples), [n_samples // n_classes, 2 * (n_samples // n_classes)])

# Generate informative features for each label
# Label A: rel1, rel2, rel3, rel4, rel5
X[split_indices[0], :5] = np.random.normal(loc=1.0, scale=1.0, size=(len(split_indices[0]), 5))

X[split_indices[1], :5] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[1]), 5))
X[split_indices[2], :5] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[2]), 5))


# Label B: rel6, rel7, rel8, rel9, rel10
X[split_indices[1], 5:10] = np.random.normal(loc=3.0, scale=1.0, size=(len(split_indices[1]), 5))

X[split_indices[0], 5:10] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[0]), 5))
X[split_indices[2], 5:10] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[2]), 5))

# Label C: rel11, rel12, rel13, rel14, rel15
X[split_indices[2], 10:15] = np.random.normal(loc=2.0, scale=1.0, size=(len(split_indices[2]), 5))

X[split_indices[0], 10:15] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[0]), 5))
X[split_indices[1], 10:15] = np.random.normal(loc=0.0, scale=1.0, size=(len(split_indices[1]), 5))

# Add irrelevant features (noise)
X[:, n_informative:] = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features - n_informative))

# Create feature names
informative_features = [f"rel{i+1}" for i in range(n_informative)]  # Names for informative features
irrelevant_features = [f"irrel{i+1}" for i in range(n_features - n_informative)]  # Names for irrelevant features
feature_names = informative_features + irrelevant_features  # Combine the two lists

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)

# Assign labels
labels = np.array(["A"] * len(split_indices[0]) + ["B"] * len(split_indices[1]) + ["C"] * len(split_indices[2]))
df["target"] = labels

# Save the dataset to a .csv file
#output_file = "synthetic_dataset_label_specific_100features.csv"
#df.to_csv(output_file, index=False)

#print(f"Dataset saved to {output_file}")

# Save the original dataset to a .csv file
output_file_original = "DATA/synthetic_dif_3ABC_500.csv"
df.to_csv(output_file_original, index=False)
print(f"Original dataset saved to {output_file_original}")

# Create a copy of the dataset for modification
df_modified = df.copy()

# Replace "A" and "B" with "AB" in the target column
df_modified["target"] = df_modified["target"].replace({"A": "AB", "B": "AB"})

# Save the modified dataset to a .csv file
output_file_modified = "DATA/synthetic_dif_2ABC_500.csv"
df_modified.to_csv(output_file_modified, index=False)
print(f"Modified dataset saved to {output_file_modified}")