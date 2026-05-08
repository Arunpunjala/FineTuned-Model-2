# Flask AI Chatbot (FLAN-T5)

A modular Flask application that uses a fine-tuned **FLAN-T5** model for conversational AI, featuring sentiment-aware prompt adjustment.

## 🚀 Features
* **LLM Integration:** Powered by Hugging Face `transformers` and TensorFlow.
* **Sentiment Awareness:** Automatically detects negative user sentiment and adjusts system prompts for calmer responses.
* **Fine-tuning Script:** Includes a training pipeline using the Open-Orca dataset.
* **Industry Structure:** Clean separation between application logic and training scripts.

---

## 📂 Project Structure
* `app/`: Core Flask application (Blueprints, Inference, Sentiment logic).
* `training/`: Scripts for dataset preprocessing and model fine-tuning.
* `main.py`: The entry point to run the Flask server.
* `models/`: (Local only) Storage for fine-tuned model weights.

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🧠 Training the Model
If you want to fine-tune the model yourself:
1. Navigate to the training directory: `cd training`
2. Run the training script: `python train.py`
This will save the model to the `../models/flan_t5_finetuned` directory.

---

## 🖥️ Running the App

1. Start the Flask server:
   ```bash
   python main.py