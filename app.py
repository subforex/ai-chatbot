from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__, static_folder='static')

model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

BOT_PERSONALITY = "You are a friendly and helpful assistant."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    prompt = f"{BOT_PERSONALITY} User: {user_message} Assistant:"
    response = chatbot(prompt, max_length=100, pad_token_id=50256)
    reply = response[0]['generated_text'].split("Assistant:")[-1].strip()
    return jsonify({"reply": reply})

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
