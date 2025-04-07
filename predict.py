from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./results/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def predict_toxicity(text):
    result = classifier(text)[0]
    label = "Toxic" if result["label"] == "LABEL_1" else "Non-toxic"
    confidence = result["score"]
    return label, confidence

def interactive_cli():
    print("\n===== Toxicity Classifier =====")
    print("Type 'quit', 'exit', or press Ctrl+C to exit")
    print("Enter text to check for toxicity:")
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            label, confidence = predict_toxicity(user_input)
            
            if label == "Toxic":
                print(f"Result: \033[91m{label}\033[0m with {confidence:.4f} confidence")
            else:
                print(f"Result: \033[92m{label}\033[0m with {confidence:.4f} confidence")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_cli()
