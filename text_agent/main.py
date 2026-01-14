# text_agent/main.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Any, cast
from pathlib import Path

# === Load fine-tuned model and tokenizer ===
model_path = Path(__file__).resolve().parent / "sentiment-analysis-model"

# Force local loading to avoid accidental Hugging Face Hub requests
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)

# === Define classification pipeline ===
nlp = pipeline("sentiment-analysis", model=cast(Any, model), tokenizer=cast(Any, tokenizer))  # type: ignore

def analyze_text(text: str):
    """
    Runs the fine-tuned DistilBERT model on input text.
    Returns the predicted label and confidence score.
    """                                 
    result = nlp(text)[0] 
    # Pad label to 10 characters for uniform JSONL storage
    # Labels: normal(6), anxiety(7), suicidal(8), depression(10)
    label_padded = result["label"].ljust(10)
    return {
        "agent": "text_agent",
        "label": label_padded,
        "score": round(result["score"], 3)
    }

# local test
if __name__ == "__main__":
    import json
    sample = "I am tired of everything and feel hopeless."
    output = analyze_text(sample)
    print(json.dumps(output, indent=2))
