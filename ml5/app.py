import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset embedded directly in the code
sample_data = {
    "gestures": [
        {"label": 0, "landmarks": [[0.5, 0.3], [0.6, 0.4], [0.4, 0.5], [0.5, 0.7], [0.3, 0.5], [0.6, 0.6], [0.7, 0.4], [0.4, 0.6], [0.5, 0.5], [0.5, 0.4]]},
        {"label": 1, "landmarks": [[0.1, 0.5], [0.2, 0.6], [0.3, 0.7], [0.3, 0.8], [0.2, 0.5], [0.1, 0.6], [0.2, 0.7], [0.3, 0.5], [0.4, 0.5], [0.2, 0.3]]},
        {"label": 0, "landmarks": [[0.4, 0.3], [0.5, 0.4], [0.3, 0.5], [0.5, 0.8], [0.3, 0.6], [0.6, 0.5], [0.7, 0.3], [0.4, 0.7], [0.5, 0.6], [0.6, 0.5]]},
        {"label": 1, "landmarks": [[0.2, 0.5], [0.3, 0.6], [0.1, 0.4], [0.3, 0.7], [0.2, 0.4], [0.1, 0.5], [0.3, 0.6], [0.2, 0.3], [0.4, 0.4], [0.1, 0.2]]},
        # More samples can be added here
    ]
}

# Load and preprocess data
def load_data(data):
    X = []
    y = []
    for gesture in data["gestures"]:
        landmarks = np.array(gesture["landmarks"])
        label = gesture["label"]
        X.append(landmarks)
        y.append(label)

    X = np.array(X).reshape(-1, 10, 2, 1)
    y = to_categorical(y)
    return X, y

# Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 1)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data from the sample dataset
X, y = load_data(sample_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = create_cnn_model(input_shape, num_classes)

# Train the model and store training history
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predictions and classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Plot Training History (Accuracy and Loss)
# Accuracy Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
