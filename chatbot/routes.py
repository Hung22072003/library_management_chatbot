from flask import Blueprint, request, jsonify
from chatbot.logic import chatbot_response

chat_blueprint = Blueprint("chat_blueprint", __name__)

@chat_blueprint.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")
    response = chatbot_response(user_input)
    return jsonify(response)
