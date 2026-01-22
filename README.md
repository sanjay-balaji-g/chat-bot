# chat-bot
# 🤖 ML-Based Chatbot using Python

A simple yet effective **Machine Learning–based chatbot** built using **Python** and **Scikit-learn**.  
This project uses **TF-IDF vectorization** and **Logistic Regression** to classify user intents and generate appropriate responses.

---

## 📌 Features

- Intent-based chatbot
- Uses **TF-IDF + Logistic Regression**
- Trains dynamically from intent patterns
- Confidence threshold handling for unknown inputs
- Logs all conversations with timestamps
- Easy to extend with new intents
- Command-line interface

---

## 🧠 How It Works

1. User input is **preprocessed** (lowercased, cleaned)
2. Input is transformed using **TF-IDF Vectorizer**
3. A **Logistic Regression** model predicts the intent
4. If confidence is low → fallback response
5. Otherwise → a random response is chosen from the matched intent
6. Chat history is saved to a log file

---

## 📂 Project Structure

chatbot/
│
├── main.py # Main chatbot logic
├── intents.ipynb # Intent definitions (patterns & responses)
├── chat_logs.txt # Conversation logs
├── requirements.txt # Required dependencies
└── README.md # Project documentation


---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- nbformat (to read intents from Jupyter Notebook)

---

## 📦 Installation

1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ How to Run
python main.py
▶️ How to Run
python main.py

---

💬 Sample Interaction
You: Hi
Bot: Hello! How can I assist you?

You: Who created you?
Bot: I was created as part of a machine learning academic project.

You: quit
Bot: Goodbye! 👋

🚀 Future Improvements

Add deep learning models (LSTM / Transformer)

Web interface using Flask or FastAPI

Context-aware conversations

Voice input/output

Model persistence (save/load trained model)

🎓 Use Case

This project is ideal for:

Machine Learning mini-projects

Academic submissions

Beginners learning NLP

Resume / portfolio projects

📜 License

This project is open-source and available for educational use.

🙌 Author

Sanjay Balaji
Machine Learning & Python Enthusiast
