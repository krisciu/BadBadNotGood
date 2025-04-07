from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
dataset = load_dataset("google/jigsaw_toxicity_pred", data_dir="datasets/toxicity")


negative_features = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def make_binary_labels(example):
    example['label'] = 1 if any(example[c] == 1 for c in negative_features) else 0
    return example

dataset = dataset.map(make_binary_labels)
dataset = dataset.remove_columns(negative_features)

# train and test

train_dataset = dataset['train']
test_dataset = dataset['test']

#TOKENIZE

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['comment_text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    push_to_hub=False,
)

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()

print(results)

trainer.save_model("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
print("Model and tokenizer saved to ./results/final_model")
