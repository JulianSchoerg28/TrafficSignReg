from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from pathlib import Path

# === Setup
working_directory = Path(__file__).resolve().parent.parent
models_dir = working_directory / "models"
image_path = working_directory / "archive" / "beispielbild.png"
image_size = 32

# === Bild laden & vorbereiten


def prepare_image(path):
    image = Image.open(path).resize((image_size, image_size))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image) / 255.0
    return image


# === Lade Bild für beide Modelle
image_for_cnn = np.expand_dims(prepare_image(
    image_path), axis=0)  # (1, 32, 32, 3)
image_for_mlp = image_for_cnn.reshape(1, -1)  # (1, 3072)

# === Modelle laden
cnn_model = load_model(models_dir / "traffic_sign_cnn_model.h5")
mlp_model = load_model(models_dir / "mlp_model.h5")

# === Vorhersagen
cnn_pred_probs = cnn_model.predict(image_for_cnn)
mlp_pred_probs = mlp_model.predict(image_for_mlp)

cnn_pred_class = np.argmax(cnn_pred_probs)
mlp_pred_class = np.argmax(mlp_pred_probs)

print("CNModell:")
print(f"  → Klasse: {cnn_pred_class} mit {100 * np.max(cnn_pred_probs):.2f}%")

print("MLModell:")
print(f"  → Klasse: {mlp_pred_class} mit {100 * np.max(mlp_pred_probs):.2f}%")
