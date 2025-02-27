from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import transformers

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
)
from transformers import TrainingArguments, Trainer

local_path = "./grover_local"  # Path where the files were copied

# Load tokenizer and model from local storage
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, texts, labels, tokenizer):

        super(SupervisedDataset, self).__init__()

        sequences = [text for text in texts]

        output = tokenizer(
            sequences,
            add_special_tokens=True,
            max_length=310,
            padding="longest",
            return_tensors="pt",
            truncation=True
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i]
        )


# Load dataset from TSV
# Update with your actual dataset path
dataset = pd.read_csv("cancer_variants.tsv", sep="\t")

# Ensure required columns exist
assert "sequence" in dataset.columns, "Missing 'sequence' column"
assert "label" in dataset.columns, "Missing 'label' column"

# 📌 Split dataset into train, validation, and test
train = dataset.sample(frac=0.6, random_state=0)
validation = dataset.drop(train.index)
test = validation.sample(frac=0.5, random_state=0)
validation = validation.drop(test.index)

# Reset index after splitting
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
test = test.reset_index(drop=True)

print(f"Train size: {len(train)} | Validation size: {len(validation)} | Test size: {len(test)}")

# Create dataset objects
train_dataset = SupervisedDataset(train["sequence"], train["label"], tokenizer)
test_dataset = SupervisedDataset(test["sequence"], test["label"], tokenizer)
val_dataset = SupervisedDataset(
    validation["sequence"], validation["label"], tokenizer)


def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": matthews_corrcoef(
            labels, predictions
        ),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

train_args = TrainingArguments(seed=42,
                               output_dir=".",
                               per_device_train_batch_size=16,
                               eval_strategy="epoch",
                               learning_rate=0.000001,
                               num_train_epochs=4
                               )

trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=train_args
)
trainer.train()

trainer.train()
results = trainer.evaluate(eval_dataset=test_dataset)
