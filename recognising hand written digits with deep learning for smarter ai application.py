import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras import layers
import uuid
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and Preprocess Data
def load_and_preprocess_data():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)
    
    # Reshape for MLP (flatten)
    X_train_mlp = X_train.reshape(-1, 784)
    X_test_mlp = X_test.reshape(-1, 784)
    
    # One-hot encode labels
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    return (X_train, X_train_cnn, X_train_mlp, y_train, y_train_cat,
            X_test, X_test_cnn, X_test_mlp, y_test, y_test_cat)

# 2. Exploratory Data Analysis
def perform_eda(X_train, y_train):
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Visualize sample digits
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f'Digit: {y_train[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/sample_digits.png')
    plt.close()
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_train)
    plt.title('Distribution of Digits in Training Set')
    plt.savefig('output/class_distribution.png')
    plt.close()
    
    # Average digit representation
    plt.figure(figsize=(12, 8))
    for digit in range(10):
        avg_img = np.mean(X_train[y_train == digit], axis=0)
        plt.subplot(2, 5, digit+1)
        plt.imshow(avg_img, cmap='gray')
        plt.title(f'Average {digit}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/average_digits.png')
    plt.close()

# 3. Build CNN Model
def build_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 4. Build MLP Model
def build_mlp_model():
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# 5. Visualize CNN Filters
def visualize_cnn_filters(model, X_test):
    # Get first layer filters
    filters = model.layers[0].get_weights()[0]
    plt.figure(figsize=(12, 8))
    for i in range(min(32, filters.shape[3])):
        plt.subplot(4, 8, i+1)
        plt.imshow(filters[:, :, 0, i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/cnn_filters.png')
    plt.close()

# 6. Plot Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

# 7. Main Execution
def main():
    # Load and preprocess data
    (X_train, X_train_cnn, X_train_mlp, y_train, y_train_cat,
     X_test, X_test_cnn, X_test_mlp, y_test, y_test_cat) = load_and_preprocess_data()
    
    # Perform EDA
    perform_eda(X_train, y_train)
    
    # Split data (using validation set)
    X_train_cnn, X_val_cnn, y_train_cat, y_val_cat = train_test_split(
        X_train_cnn, y_train_cat, test_size=0.2, random_state=42)
    X_train_mlp, X_val_mlp = train_test_split(
        X_train_mlp, test_size=0.2, random_state=42)
    
    # Build and train CNN
    cnn_model = build_cnn_model()
    cnn_history = cnn_model.fit(X_train_cnn, y_train_cat,
                              epochs=10,
                              batch_size=128,
                              validation_data=(X_val_cnn, y_val_cat),
                              verbose=1)
    
    # Build and train MLP
    mlp_model = build_mlp_model()
    mlp_history = mlp_model.fit(X_train_mlp, y_train_cat,
                              epochs=10,
                              batch_size=128,
                              validation_data=(X_val_mlp, y_val_cat),
                              verbose=1)
    
    # Evaluate models
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test_cat)
    mlp_test_loss, mlp_test_acc = mlp_model.evaluate(X_test_mlp, y_test_cat)
    
    print("\nModel Performance:")
    print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
    print(f"MLP Test Accuracy: {mlp_test_acc:.4f}")
    
    # Generate predictions
    cnn_pred = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    mlp_pred = np.argmax(mlp_model.predict(X_test_mlp), axis=1)
    
    # Classification reports
    print("\nCNN Classification Report:")
    print(classification_report(y_test, cnn_pred))
    print("\nMLP Classification Report:")
    print(classification_report(y_test, mlp_pred))
    
    # Visualize results
    plot_confusion_matrix(y_test, cnn_pred, 'CNN Confusion Matrix',
                         'output/cnn_confusion_matrix.png')
    plot_confusion_matrix(y_test, mlp_pred, 'MLP Confusion Matrix',
                         'output/mlp_confusion_matrix.png')
    
    visualize_cnn_filters(cnn_model, X_test)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'], label='CNN Train')
    plt.plot(cnn_history.history['val_accuracy'], label='CNN Val')
    plt.plot(mlp_history.history['accuracy'], label='MLP Train')
    plt.plot(mlp_history.history['val_accuracy'], label='MLP Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['loss'], label='CNN Train')
    plt.plot(cnn_history.history['val_loss'], label='CNN Val')
    plt.plot(mlp_history.history['loss'], label='MLP Train')
    plt.plot(mlp_history.history['val_loss'], label='MLP Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/training_history.png')
    plt.close()

if _name_ == '_main_':
    main()
