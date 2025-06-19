from gpt4all import GPT4All

model = GPT4All(
    model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    model_path="/home/lenz/.local/share/nomic.ai/GPT4All"
)

print("Starte Chat. Tippe 'exit' zum Beenden.")

with model.chat_session() as session:
    while True:
        frage = input("Du: ")
        if frage.lower() == "exit":
            break
        antwort = session.chat(frage)  # <-- changed here
        print("GPT4All:", antwort)
