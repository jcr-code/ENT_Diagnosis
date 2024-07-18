from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from RandomForest_FeatureRank import RandomForest
from sklearn.preprocessing import LabelEncoder

# Load your dataset from the Excel file
file_path = "../Progress_Proposal_THT/data_THT_transform.xlsx"
df = pd.read_excel(file_path)

# Drop rows with missing values
df.dropna(axis=0, inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the target labels into numeric values
df['hasil_diagn_encoded'] = label_encoder.fit_transform(df['hasil_diagn'])

# Check the mapping between original labels and encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# print("Label Mapping:", label_mapping)

# Now you have a new column 'hasil_diagn_encoded' containing numeric representations of the labels
# You can drop the original 'hasil_diagn' column if you don't need it anymore
df.drop(columns=['hasil_diagn'], inplace=True)

# Example: Inverse transform to get original labels from encoded values
# original_labels = label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14])  # Replace [0, 1, 2] with your encoded values
# print("Original Labels:", original_labels)

# Assuming the last column is the label/target column
X = df.drop("hasil_diagn_encoded", axis = 1).values  # Features + converting into numpy array
y = df['hasil_diagn_encoded'].values   # Labels + converting into numpy array

# # Assuming the last column is the label/target column
# X = df.iloc[:, :-1]  # Features
# y = df.iloc[:, -1]   # Labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42 #1234
)



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)