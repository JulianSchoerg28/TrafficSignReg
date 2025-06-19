from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from pathlib import Path

working_directory = Path(__file__).resolve().parent.parent

# Modell laden
model = load_model(working_directory / "traffic_sign_cnn_model.h5")

# Bild laden und vorbereiten
image_path = working_directory / "archive" / "beispielbild.png"
image_size = 32
image = Image.open(image_path).resize((image_size, image_size))
image = np.array(image)
if image.shape != (image_size, image_size, 3):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Vorhersage
pred_probs = model.predict(image)
pred_class = np.argmax(pred_probs)
print(f"Vorhergesagte Klasse: {pred_class}")
