import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import tensorflow as tf

from pathlib import Path

working_directory = Path(__file__).resolve().parent.parent
# create dir models if not exists
output_dir = working_directory / "models"
output_dir.mkdir(parents=True, exist_ok=True)

print("TensorFlow-Version:", tf.__version__)
print("GPU verfügbar:", tf.config.list_physical_devices('GPU'))

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# === Parameters ===
num_classes = 43
image_size = 32
dataset_path = working_directory / "archive" / "train"

# === Load and preprocess data ===
data = []
labels = []

print("Loading images...")
for class_id in range(num_classes):
    class_path = os.path.join(dataset_path, str(class_id))
    for img_name in os.listdir(class_path):
        try:
            img_path = os.path.join(class_path, img_name)
            image = Image.open(img_path)
            image = image.resize((image_size, image_size))
            image = np.array(image)
            if image.shape == (image_size, image_size, 3):
                data.append(image)
                labels.append(class_id)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")

data = np.array(data)
labels = np.array(labels)
print(f"Loaded {data.shape[0]} images.")

# === Normalize data ===
data = data / 255.0

# === Split the data ===
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels)

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# === Define the CNN model ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(image_size, image_size, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# === Train the model ===
history = model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=15,
    validation_split=0.2,
    verbose=1
)

# === Evaluate the model ===
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {accuracy:.4f}")

# === Predictions and confusion matrix ===
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(working_directory / "models" / "confusion_matrix.png")

# === Plot training history ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(working_directory / "models" / "accuracy_plot.png")

# === Save the model ===
model.save(working_directory / "models" / "traffic_sign_cnn_model.h5")
print("Model saved as traffic_sign_cnn_model.h5")
