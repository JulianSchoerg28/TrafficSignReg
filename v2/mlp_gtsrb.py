import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from pathlib import Path

# === Setup ===
working_dir = Path(__file__).resolve().parent.parent
data_dir = working_dir / "archive" / "train"
output_dir = working_dir / "models"
output_dir.mkdir(exist_ok=True)

num_classes = 43
image_size = 32

# === Bilder laden ===
images = []
labels = []

print("Lade Bilder...")
for class_id in range(num_classes):
    class_path = data_dir / str(class_id)
    for img_name in os.listdir(class_path):
        try:
            img_path = class_path / img_name
            image = Image.open(img_path).resize((image_size, image_size))
            image = np.array(image)
            if image.shape == (image_size, image_size, 3):
                images.append(image)
                labels.append(class_id)
        except Exception as e:
            print(f"Fehler bei {img_name}: {e}")

images = np.array(images) / 255.0
labels = np.array(labels)
print(f"{len(images)} Bilder geladen.")

# === Flatten für MLP
X = images.reshape((images.shape[0], -1))  # z. B. (39209, 3072)
y = to_categorical(labels, num_classes)

# === Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=labels, random_state=42)

# === MLP Modell
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Training
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# === Evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test-Accuracy: {acc:.4f}")

# === Vorhersagen & Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (MLP)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix_mlp.png")

# === Plots speichern
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.tight_layout()
plt.savefig(output_dir / "accuracy_plot_mlp.png")

# === Modell speichern
model.save(output_dir / "mlp_model.h5")
print("Modell gespeichert als mlp_model.h5")
