# Optional: automatische Installation der nötigen Pakete (einmalig)
try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
except ImportError:
    import os
    os.system("pip install llama-cpp-python huggingface-hub")
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download

import os

# Schritt 1: Modell herunterladen (nur wenn nicht vorhanden)
model_file = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
if not os.path.exists(model_file):
    print("📥 Lade Modell herunter...")
    model_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        local_dir="models",
        local_dir_use_symlinks=False
    )
else:
    print("✅ Modell bereits vorhanden.")
    model_path = model_file

# Schritt 2: LLM initialisieren
print("🚀 Lade Modell in Speicher...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
)

# Schritt 3: Interaktive Konsolenschleife
print("\n💬 Du kannst jetzt Fragen stellen (oder 'exit' zum Beenden):\n")

while True:
    prompt = input("Erzähl mir kurz etwas über: ")
    if prompt.strip().lower() in ["exit", "quit"]:
        print("👋 Auf Wiedersehen!")
        break

    response = llm(prompt, max_tokens=300)  # ohne stop!
    answer = response["choices"][0]["text"].strip()

    if not answer:
        print("🧠 Antwort: (keine Ausgabe vom Modell)")
    else:
        print("🧠 Antwort:\n" + answer + "\n")
