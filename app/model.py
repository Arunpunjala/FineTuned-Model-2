from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
# Ensure your config.py exists and contains MODEL_PATH
from config import MODEL_PATH 

# Load model and tokenizer
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def generate_response(prompt):
    # Convert prompt text into input IDs
    inputs = tokenizer(
        prompt,
        return_tensors="tf",
        max_length=1024,
        truncation=True,
        padding=True
    )

    # Generate output tokens
    outputs = model.generate(
        inputs["input_ids"],
        max_length=250
    )

    # Convert output IDs back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response