import tensorflow as tf
from datasets import load_dataset
from transformers import TFAutoModelForSeq2SeqLM
from preprocess import preprocess_function

# Load and prepare dataset
dataset = load_dataset("Open-Orca/OpenOrca")
dataset = dataset["train"].select(range(100))

tokenized = dataset.map(preprocess_function, batched=True)
tokenized.set_format("tensorflow")

# Load model
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Convert to TensorFlow dataset
tf_dataset = tokenized.to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels", "decoder_input_ids"],
    shuffle=True,
    batch_size=16,
    drop_remainder=True
)

# Train
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(tf_dataset, epochs=2)

# Save
model.save_pretrained("../models/flan_t5_finetuned")