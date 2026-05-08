from flask import Blueprint, request, jsonify
from .model import generate_response
from .sentiment import adjust_prompt

chat_bp = Blueprint("chat", __name__)

# Simple conversation history (in-memory)
conversation_history = []

@chat_bp.route("/chat", methods=["POST"])
def chat():
    # Get user input from request JSON
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Add user input to history
    conversation_history.append(f"User: {user_input}")

    # Adjust input using sentiment
    adjusted_prompt = adjust_prompt(user_input)

    # Create full conversation context
    full_prompt = "\n".join(conversation_history + [adjusted_prompt])

    # Generate response
    response_text = generate_response(full_prompt)
    
    # Append model response to history
    conversation_history.append(f"Bot: {response_text}")

    return jsonify({"response": response_text})