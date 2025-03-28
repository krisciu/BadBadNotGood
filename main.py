from datasets import load_dataset

from transformers import AutoTokenizer

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


print(train_dataset)
print(test_dataset)

