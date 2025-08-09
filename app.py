from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# You can change the model here (smaller models are faster and free to use)
chatbot = pipeline("text-generation", model="sshleifer/tiny-gpt2")

# Optional: customise your bot's style or domain
BOT_PERSONALITY = "You are a friendly and helpful assistant."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    # Combine personality + user message
    prompt = f"{BOT_PERSONALITY} User: {user_message} Assistant:"
    response = chatbot(prompt, max_length=200, pad_token_id=50256)

    return jsonify({"reply": response[0]['generated_text'].split("Assistant:")[-1].strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
