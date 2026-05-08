from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def adjust_prompt(prompt):
    sentiment = sentiment_pipeline(prompt)[0]

    # Correct/adjust if negative
    if sentiment["label"].lower() == "negative":
        return f"User is upset. Respond calmly: {prompt}"
        
    return prompt