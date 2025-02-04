# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:10:54 2025

@author: Marwa
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import tkinter as tk
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset
data = pd.read_csv('./WA.csv')

# Preprocessing: Handle missing values and convert categorical features to numerical
data.dropna(inplace=True)  # Drop rows with missing values
print(f"Data after dropping missing values: {data.shape}")

# Handle categorical columns by encoding them
label_encoder = LabelEncoder()

# Apply label encoding to the target column ('class') and other categorical columns
data['Class'] = label_encoder.fit_transform(data['Class'])  # Target column

# Convert other categorical variables to numerical (if needed)
X = data.drop('Class', axis=1)
X = pd.get_dummies(X)  # One-hot encode categorical features
y = data['Class']

# Balancing the dataset using RandomOverSampler (instead of SMOTE)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Feature Selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)

# Get the names of the selected features
selected_features = X.columns[selector.get_support()]

# Features Before and After Feature Selection
features_before = X.columns
features_after = selected_features

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_resampled, test_size=0.2, random_state=42)

# Deep Learning Model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Evaluate the model before training
y_pred_initial = model.predict(X_test)
y_pred_class_initial = (y_pred_initial > 0.5).astype("int32")  # Convert predictions to binary class (0 or 1)
accuracy_initial = accuracy_score(y_test, y_pred_class_initial)
auc_initial = roc_auc_score(y_test, y_pred_initial)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# Evaluate the model after training
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred_class)
auc = roc_auc_score(y_test, y_pred)

# Print results
print(f"Initial Accuracy: {accuracy_initial:.4f}")
print(f"Initial AUC: {auc_initial:.4f}")
print(f"Trained Accuracy: {accuracy:.4f}")
print(f"Trained AUC: {auc:.4f}")

# Plot Accuracy and AUC before and after training
metrics = ['Accuracy', 'AUC']
initial_metrics = [accuracy_initial, auc_initial]
trained_metrics = [accuracy, auc]

# Create a bar plot for accuracy and AUC comparison
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
index = np.arange(len(metrics))

bar1 = ax.bar(index, initial_metrics, bar_width, label='Before Training')
bar2 = ax.bar(index + bar_width, trained_metrics, bar_width, label='After Training')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation: Before and After Training')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Balancing Before and After using RandomOverSampler
# Visualize the class distribution before and after balancing
plt.figure(figsize=(8, 5))

# Before balancing
plt.subplot(1, 2, 1)
plt.title('Class Distribution Before Balancing')
plt.hist(y, bins=10, color='blue', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Count')

# After balancing
plt.subplot(1, 2, 2)
plt.title('Class Distribution After Balancing')
plt.hist(y_resampled, bins=10, color='green', alpha=0.7)
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Display Features Before and After Feature Selection
print("Features before feature selection:")
print(features_before)
print("\nFeatures after feature selection:")
print(features_after)

# Tkinter GUI to accept user input in one window with selected features
def get_user_input():
    root = tk.Tk()
    root.title("Enter Feature Values (Selected Features)")

    # Create labels and input fields for each selected feature
    input_data = []
    
    # Create a label and entry field for each selected feature
    for i, feature in enumerate(selected_features):
        label = tk.Label(root, text=f"Enter value for {feature}:")
        label.grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1)
        input_data.append(entry)
    
    # Function to handle the submission and prediction
    def submit():
        values = []
        for entry in input_data:
            values.append(float(entry.get()) if entry.get() else 0)  # Convert to float (default to 0 if empty)
        input_array = np.array(values).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(input_array)
        prediction_class = (prediction > 0.5).astype("int32")

        result_label.config(text=f"Prediction result: {'Positive' if prediction_class[0][0] == 1 else 'Negative'}")

    # Button to submit the inputs and make prediction
    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=len(selected_features), column=1)

    # Label for displaying the prediction result
    result_label = tk.Label(root, text="Prediction result will appear here.")
    result_label.grid(row=len(selected_features)+1, column=1)

    # Run the Tkinter event loop
    root.mainloop()

# Run the Tkinter window to accept user input and make predictions
get_user_input()
