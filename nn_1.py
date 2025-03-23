import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset function
def load_dt_st():
    dt = pd.read_csv("mnist_train.csv")
    y = dt['label'].values
    X = dt.drop(columns=['label']).values
    print(f"Following dataset has : {len(X)} examples")

    # Normalize input features
    X = X / 255.0

    # Split the dataset into training and testing
    x = int(input("Enter the number of training examples you want of all : "))
    X_train, y_train = X[:x], y[:x]
    X_test, y_test = X[x:], y[x:]

    print(f"Test data examples  {len(y_test)} and train {len(y_train)} examples")
    return X_train, y_train, X_test, y_test

# One-hot encoding function
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

# Build and compile the model using TensorFlow/Keras
def build_model(input_shape, num_classes, layers, neurons_per_layer):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for neurons in neurons_per_layer:
        model.add(tf.keras.layers.Dense(neurons, activation='sigmoid'))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Training and evaluation function
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs):
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    # Plot accuracy and loss over epochs
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.show()

# Main function to run the program
if __name__ == "__main__":
    # Load dataset
    X_train, y_train, X_test, y_test = load_dt_st()

    # Encode labels as one-hot vectors
    uq_el = set(y_train)
    ot_cls = len(uq_el)
    print("THE NUMBER OF DIFFERENT CLASSES FOLLOWING DATASET HAS : ", ot_cls)
    y_train_encoded = one_hot_encode(y_train, ot_cls)
    y_test_encoded = one_hot_encode(y_test, ot_cls)

    # Get user input for the number of layers and neurons
    num_layers = int(input("Enter number of hidden layers: ")) + 2
    neurons_per_layer = list(map(int, input("Enter number of neurons in each hidden layer (space-separated): ").split()))

    # Build the model
    model = build_model(input_shape=(X_train.shape[1],), num_classes=ot_cls, layers=num_layers, neurons_per_layer=neurons_per_layer)

    # Get number of epochs and train the model
    epochs = int(input("Enter the number of epochs you want : "))
    train_and_evaluate(model, X_train, y_train_encoded, X_test, y_test_encoded, epochs)

    # Final evaluation on training data
    train_loss, train_acc = model.evaluate(X_train, y_train_encoded)
    print(f"Final Training accuracy: {train_acc * 100:.2f}%")

    # Final evaluation on test data
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
    print(f"Test accuracy: {test_acc * 100:.2f}%")
