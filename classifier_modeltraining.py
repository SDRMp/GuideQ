import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Define dataset type and model name
dstype = 'stress'
mname = 'debertaV3'

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', f'{dstype}.csv')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models', f'{mname}_{dstype}')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
df['label'] = df['subreddit']

# Encode labels
le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])

# Visualize label distribution
plt.figure(figsize=(8,6))
df.groupby('label').text.count().sort_values().plot.barh(title='NUMBER OF TEXTS IN EACH CATEGORY')
plt.xlabel('Number of occurrences')
plt.show()

# Tokenizer and Model Loading
MODEL_PATH = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
num_labels = df['target'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Token length analysis
df['token_length'] = df['text'].apply(lambda x: len(tokenizer.tokenize(str(x))))
print(f"Max token length: {df['token_length'].max()}")
print(f"Avg token length: {df['token_length'].mean():.2f}")

# Data Splitting
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
for train_val_idx, test_idx in sss.split(df, df['target']):
    train_val_df, test_df = df.iloc[train_val_idx], df.iloc[test_idx]

sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
for train_idx, val_idx in sss_val.split(train_val_df, train_val_df['target']):
    train_df, val_df = train_val_df.iloc[train_idx], train_val_df.iloc[val_idx]

# Tokenization function
def tokenize_and_format(examples):
    tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    tokenized_inputs['label'] = list(map(int, examples['target']))
    return tokenized_inputs

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df).map(tokenize_and_format, batched=True)
eval_dataset = Dataset.from_pandas(val_df).map(tokenize_and_format, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_and_format, batched=True)

# Define metrics
def compute_metrics(pred):
    labels, preds = pred.label_ids, pred.predictions.argmax(-1)
    return {
        'eval_f1': f1_score(labels, preds, average='weighted'),
        'eval_precision': precision_score(labels, preds, average='weighted'),
        'eval_recall': recall_score(labels, preds, average='weighted'),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    logging_dir=LOGS_DIR,
    logging_steps=10,
    report_to=[]
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)

# Evaluate model
def evaluate_model(dataset, dataset_name):
    print(f"Evaluating on {dataset_name} dataset...")
    predictions, labels, _ = trainer.predict(dataset)
    predictions = np.argmax(predictions, axis=1)
    print(classification_report(labels, predictions, target_names=le.classes_, digits=4))

evaluate_model(eval_dataset, "Validation")
evaluate_model(test_dataset, "Test")