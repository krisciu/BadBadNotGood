from datasets import load_dataset

dataset = load_dataset("google/jigsaw_toxicity_pred", data_dir="datasets/toxicity")

print(dataset)

