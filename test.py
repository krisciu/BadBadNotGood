from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./results/final_model")
tokenizer = AutoTokenizer.from_pretrained("./results/final_model")

# Create a pipeline for easy inference
from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test it
text = "I hate you!!!"
result = classifier(text)
print(result)
