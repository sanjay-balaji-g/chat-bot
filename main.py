import json
import random
import re
import datetime
import numpy as np
import nbformat

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Text Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# -------------------------------
# Load Intent Data
# -------------------------------

def load_intents_from_ipynb(path):
    nb = nbformat.read(path, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == "code":
            local_vars = {}
            exec(cell.source, {}, local_vars)

            if "intents" in local_vars:
                return local_vars["intents"]

    raise ValueError("intents variable not found in notebook")

data = load_intents_from_ipynb("intents.ipynb")


sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(preprocess(pattern))
        labels.append(intent["tag"])

# -------------------------------
# Feature Extraction
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(sentences)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# -------------------------------
# Logging Function
# -------------------------------
def log_chat(user, bot):
    with open("chat_logs.txt", "a") as file:
        time = datetime.datetime.now()
        file.write(f"[{time}] User: {user}\n")
        file.write(f"[{time}] Bot: {bot}\n")

# -------------------------------
# Context Memory
# -------------------------------
last_intent = None

# -------------------------------
# Response Generator
# -------------------------------
def get_bot_response(user_input):
    global last_intent

    processed = preprocess(user_input)
    vector = vectorizer.transform([processed])

    probs = model.predict_proba(vector)[0]
    confidence = np.max(probs)
    predicted_intent = model.classes_[np.argmax(probs)]

    # Confidence threshold
    if confidence < 0.25:
        response = "I'm not sure I understood that. Could you please rephrase?"
        log_chat(user_input, response)
        return response

    # Store context
    last_intent = predicted_intent

    for intent in data["intents"]:
        if intent["tag"] == predicted_intent:
            response = random.choice(intent["responses"])
            log_chat(user_input, response)
            return response

# -------------------------------
# Main Chat Loop
# -------------------------------
print("\n🤖 Advanced ML Chatbot Activated")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    reply = get_bot_response(user_input)
    print("Bot:", reply)
