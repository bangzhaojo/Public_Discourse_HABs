import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


def list_of_dicts_to_dataset(list_of_dicts):
    # Initialize a dictionary to hold our reformatted data
    data_formatted = {key: [] for key in list_of_dicts[0].keys()}
    
    # Populate the dictionary with data from list_of_dicts
    for item in list_of_dicts:
        for key, value in item.items():
            data_formatted[key].append(value)
            
    # Now data_formatted is in the correct format for from_dict
    return Dataset.from_dict(data_formatted)


with open('/shared/3/projects/bangzhao/lake-erie/data_cleaned/gpt3.5_10000.json', 'r') as json_file:
    data = json.load(json_file)
    
dataset = [{'text':item[0], 'label':item[1]} for item in data]
dataset = [{'text': item['text'], 'label': 1 if item['label'] == 'True' else 0} for item in dataset]
train_set, validation_set = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataset = list_of_dicts_to_dataset(train_set)
validation_dataset = list_of_dicts_to_dataset(validation_set)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=3)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = validation_dataset.map(tokenize_function, batched=True)

def compute_metrics(pred):
    # pred.label_ids are the true labels
    # pred.predictions are the logits from the model
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # Convert logits to class predictions

    # Calculate accuracy and F1 score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')  # Use 'micro', 'macro', or 'weighted' for multi-class
    
    return {
        'accuracy': acc,
        'f1': f1,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='/shared/3/projects/bangzhao/lake-erie/models',          # output directory for checkpoints
    num_train_epochs=6,              # number of training epochs
    per_device_train_batch_size=32,   # batch size per device during training
    per_device_eval_batch_size=32,    # batch size for evaluation
    warmup_steps=50,                # number of warmup steps for learning rate scheduler
    weight_decay=0.001,               # strength of weight decay
    logging_dir='/shared/3/projects/bangzhao/lake-erie/logs',            # directory for storing logs
    logging_steps=len(train_dataset)//32,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
trainer.evaluate()

model_path = "/shared/3/projects/bangzhao/lake-erie/models/bert-large-fine-tuned"

# Save the model and the tokenizer
model.save_pretrained(model_path)