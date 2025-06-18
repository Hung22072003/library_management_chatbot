from flask import Flask
from flask_cors import CORS
from chatbot.routes import chat_blueprint

app = Flask(__name__)
app.register_blueprint(chat_blueprint)
CORS(app)

if __name__ == "__main__":
    app.run(debug=True)
