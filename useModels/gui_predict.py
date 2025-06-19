import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.models import load_model

# === Setup
current_dir = Path(__file__).parent
working_dir = current_dir.parent
models_dir = working_dir / "models"
cnn_model = load_model(models_dir / "traffic_sign_cnn_model.h5")
mlp_model = load_model(models_dir / "mlp_model.h5")
image_size = 32

with open(current_dir / "class_mapping_de.json", "r", encoding="utf-8") as f:
    class_map = json.load(f)

# === Bild vorbereiten
def prepare_image(path):
    image = Image.open(path).resize((image_size, image_size))
    if image.mode != "RGB":
        image = image.convert("RGB")
    norm = np.array(image) / 255.0
    return image, np.expand_dims(norm, axis=0), norm.reshape(1, -1)

# === Vorhersagefunktion
def predict_image(img_path):
    try:
        pil_img, cnn_input, mlp_input = prepare_image(img_path)

        pred_cnn = cnn_model.predict(cnn_input)
        pred_mlp = mlp_model.predict(mlp_input)

        class_cnn = np.argmax(pred_cnn)
        class_mlp = np.argmax(pred_mlp)

        label_cnn = class_map[str(class_cnn)]
        label_mlp = class_map[str(class_mlp)]

        prob_cnn = 100 * np.max(pred_cnn)
        prob_mlp = 100 * np.max(pred_mlp)

        result_text.set(
            f"CNN → Klasse {class_cnn}: {label_cnn} ({prob_cnn:.2f}%)\n"
            f"MLP → Klasse {class_mlp}: {label_mlp} ({prob_mlp:.2f}%)"
        )

        # Bild anzeigen
        tk_img = ImageTk.PhotoImage(pil_img.resize((150, 150)))
        image_label.config(image=tk_img)
        image_label.image = tk_img

    except Exception as e:
        messagebox.showerror("Fehler", str(e))

# === Datei auswählen
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.png *.jpg *.jpeg")])
    if file_path:
        predict_image(file_path)

# === GUI aufbauen
root = tk.Tk()
root.title("Verkehrsschilder-Vorhersage")
root.geometry("500x400")
root.resizable(False, False)

tk.Label(root, text="Wähle ein Bild aus:", font=("Arial", 14)).pack(pady=10)
tk.Button(root, text="Bild auswählen", command=open_image, font=("Arial", 12)).pack()

image_label = tk.Label(root)
image_label.pack(pady=10)

result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, font=("Arial", 12), justify="center").pack(pady=10)

root.mainloop()
