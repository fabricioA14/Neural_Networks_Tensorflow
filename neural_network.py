
# Install Python packages
import os
import subprocess

packages = ["tensorflow", "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]

# Install packages using subprocess
for package in packages:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", package])

# Load the dataset in Python environment
import pandas as pd

dataset = pd.read_csv('Species.csv')

# Let's see the head of the six first columns of our fish dataset
print(dataset.iloc[:, :6].head())

# Select only the functional characters to insert in the neural network
X = dataset.iloc[:, 1:29].values

# Wrangling and select only the labels (species names) to insert in the neural network
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Assuming y contains the names of the three classes
y_n = dataset.iloc[:, 0].values

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform the target labels
y = label_encoder.fit_transform(y_n)

# Data normalization (z-scores) of functional characters
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Separate the dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Neural network assembly and training
import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=15, activation='relu', input_shape=(28,)),
    tf.keras.layers.Dense(units=15, activation='relu'),
    tf.keras.layers.Dense(units=15, activation='relu'),
    tf.keras.layers.Dense(units=np.max(y) + 1, activation='softmax')
])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

# Neural network evaluation
import matplotlib.pyplot as plt

loss = history.history['val_loss']
accuracy = history.history['val_accuracy']
epochs = range(1, 51)

# Plot the evaluation metrics
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Validation loss')
plt.title('Validation loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo', label='Validation accuracy')
plt.title('Validation accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
