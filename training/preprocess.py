import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def preprocess_function(examples):
    # Prepare inputs
    inputs = ["question: " + q for q in examples["question"]]
    targets = [r if len(r) > 0 else "I don't know" for r in examples["response"]]

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=256, truncation=True, padding=True, return_tensors="np"
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=256, truncation=True, padding=True, return_tensors="np"
        )

    model_inputs["labels"] = labels["input_ids"].astype(np.int32)
    model_inputs["decoder_input_ids"] = labels["input_ids"].astype(np.int32)

    return model_inputs