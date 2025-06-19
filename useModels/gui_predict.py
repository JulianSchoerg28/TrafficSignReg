import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from llama_cpp import Llama

# === Setup
current_dir = Path(__file__).parent
working_directory = current_dir.parent
models_dir = working_directory / "models"
cnn_model = load_model(models_dir / "traffic_sign_cnn_model.h5")
mlp_model = load_model(models_dir / "mlp_model.h5")
llama_model_path = models_dir / "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm = Llama(model_path=str(llama_model_path), n_ctx=512)

image_size = 32
current_sign_label = ""

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
    global current_sign_label
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

        current_sign_label = f"{label_cnn} Schild"
        chat_output.delete("1.0", tk.END)

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

# === Chat absenden
def send_chat():
    user_input = chat_input.get("1.0", tk.END).strip()
    if user_input:
        chat_output.insert(tk.END, f"Du: {user_input}\n")
        chat_input.delete("1.0", tk.END)

        combined_prompt = f"Frage zum {current_sign_label}: {user_input}"
        output = llm(f"[INST] {combined_prompt} [/INST]", stop=["</s>"], max_tokens=500)
        response = output["choices"][0]["text"].strip()

        chat_output.insert(tk.END, f"KI: {response}\n\n")
        chat_output.see(tk.END)

# === GUI aufbauen
root = tk.Tk()
root.title("Verkehrsschilder + KI-Chat")
root.geometry("700x600")
root.resizable(False, False)

frame_top = tk.Frame(root)
frame_top.pack(pady=10)

btn = tk.Button(frame_top, text="Bild auswählen", command=open_image, font=("Arial", 12))
btn.pack()

image_label = tk.Label(root)
image_label.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12), justify="center")
result_label.pack(pady=10)

# Chatbereich
chat_output = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD, font=("Arial", 10))
chat_output.pack(padx=10, pady=5, fill="both")

chat_input = tk.Text(root, height=2, font=("Arial", 10))
chat_input.pack(padx=10, pady=(0, 5), fill="x")

send_button = tk.Button(root, text="Absenden", command=send_chat)
send_button.pack(pady=(0, 10))

root.mainloop()