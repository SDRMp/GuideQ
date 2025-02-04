#!/usr/bin/env python
 

import os
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
from nltk.util import ngrams
import nltk

# Ensure NLTK resources are downloaded
nltk.download("punkt")

# Configuration
class Config:
    # Experiment parameters
    NGRAM_LENGTH = 1
    DATASET_TYPE = "stress"
    MODEL_NAME = "debertaV3"
    TOP_N = 5
    
    # Path configurations
    BASE_DIR = Path("/home/bhairavi/om")
    DATA_DIR = BASE_DIR / "om4/stress"
    OUTPUT_DIR = BASE_DIR / f"om3/{DATASET_TYPE}/{NGRAM_LENGTH}grams_{MODEL_NAME}"
    MODEL_DIR = BASE_DIR / f"om5/{DATASET_TYPE}/{MODEL_NAME}_{DATASET_TYPE}"
    
    # File names
    OUTPUT_CSV = OUTPUT_DIR / f"{DATASET_TYPE}_{NGRAM_LENGTH}keys.csv"
    FULL_ANALYSIS_CSV = OUTPUT_DIR / f"{DATASET_TYPE}_{NGRAM_LENGTH}top5.csv"
    
    # Model training parameters
    TRAINING_ARGS = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
        report_to=[],
    )


def setup_environment(config: Config) -> None:
    """Create directories and set up logging"""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(config.OUTPUT_DIR / "processing.log"), logging.StreamHandler()],
    )


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and preprocess dataset"""
    df = pd.read_csv(data_path / "Stress.csv")
    df["text"] = df["text"].str.slice(0, 512)  # Truncate long texts
    return df


def prepare_datasets(df: pd.DataFrame, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """Split data into train/val/test sets and tokenize"""
    # Label encoding
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["subreddit"])
    
    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    train_val_idx, test_idx = next(sss.split(df, df["target"]))
    train_val_df, test_df = df.iloc[train_val_idx], df.iloc[test_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss_val.split(train_val_df, train_val_df["target"]))
    train_df, val_df = train_val_df.iloc[train_idx], train_val_df.iloc[val_idx]

    # Dataset preparation
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    return (
        Dataset.from_pandas(train_df).map(tokenize, batched=True, batch_size=16),
        Dataset.from_pandas(val_df).map(tokenize, batched=True, batch_size=16),
        Dataset.from_pandas(test_df).map(tokenize, batched=True, batch_size=16),
    )


class NGramAnalyzer:
    """Handles n-gram analysis for model explanations"""
    
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def analyze_ngrams(self, text: str, true_label: int, ngram_length: int) -> Dict[str, float]:
        """Perform occlusion analysis for n-gram importance"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            original_logits = self.model(**inputs).logits
            original_prob = torch.softmax(original_logits, dim=-1)[0][true_label].item()

            words = text.split()
            ngram_importances = {}
            
            for ngram in ngrams(words, ngram_length):
                occluded_text = " ".join(
                    word if word not in ngram else "[OCCLUDED]" for word in words
                )
                occluded_inputs = self.tokenizer(occluded_text, return_tensors="pt").to(self.device)
                occluded_prob = torch.softmax(self.model(**occluded_inputs).logits, dim=-1)[0][true_label].item()
                ngram_importances[" ".join(ngram)] = original_prob - occluded_prob

            return ngram_importances
def perform_analysis(test_dataset: Dataset, model: AutoModelForSequenceClassification, 
                    tokenizer: AutoTokenizer, config: Config) -> None:
    """Complete analysis pipeline including occlusion and partial text evaluation"""
    # Convert test dataset to pandas DataFrame
    test_df = pd.DataFrame(test_dataset)
    le = LabelEncoder().fit(test_df["subreddit"])
    
    # Initialize NGram analyzer
    analyzer = NGramAnalyzer(model, tokenizer, device="cuda")
    
    # Perform occlusion analysis
    logging.info("Performing occlusion analysis...")
    test_df["significant_ngrams"] = ""
    test_df["ngram_weights"] = ""

    for idx, row in test_df.iterrows():
        if idx % 100 == 0:
            logging.info(f"Processing {idx+1}/{len(test_df)} samples...")
            
        if row["target"] == row["predicted_label"]:
            try:
                ngram_importances = analyzer.analyze_ngrams(
                    row["text"], row["target"], config.NGRAM_LENGTH
                )
                positive_ngrams = {
                    k: v for k, v in ngram_importances.items() if v > 0
                }
                sorted_ngrams = sorted(positive_ngrams.items(), 
                                     key=lambda x: x[1], reverse=True)[:config.TOP_N]
                
                test_df.at[idx, "significant_ngrams"] = [k for k, v in sorted_ngrams]
                test_df.at[idx, "ngram_weights"] = [v for k, v in sorted_ngrams]

            except Exception as e:
                logging.error(f"Error processing row {idx}: {str(e)}")
                continue

    # Save intermediate results
    test_df.to_csv(config.FULL_ANALYSIS_CSV, index=False)
    logging.info(f"Saved full analysis results to {config.FULL_ANALYSIS_CSV}")

    # Aggregate significant n-grams per class
    logging.info("Aggregating n-grams per class...")
    label_to_ngrams = defaultdict(list)
    for _, row in test_df.iterrows():
        if row["significant_ngrams"]:
            label = le.inverse_transform([row["target"]])[0]
            label_to_ngrams[label].extend(zip(
                row["significant_ngrams"], 
                row["ngram_weights"]
            ))

    # Process and save top n-grams
    top_ngrams_df = pd.DataFrame(columns=["label", "ngram", "weight", "count"])
    for label, ngrams_list in label_to_ngrams.items():
        counter = defaultdict(lambda: {"weight": 0, "count": 0})
        for ngram, weight in ngrams_list:
            counter[ngram]["weight"] += weight
            counter[ngram]["count"] += 1

        sorted_ngrams = sorted(
            counter.items(), 
            key=lambda x: x[1]["weight"], 
            reverse=True
        )[:config.TOP_N]

        for ngram, stats in sorted_ngrams:
            top_ngrams_df = pd.concat([
                top_ngrams_df,
                pd.DataFrame([{
                    "label": label,
                    "ngram": ngram,
                    "weight": stats["weight"],
                    "count": stats["count"]
                }])
            ], ignore_index=True)

    top_ngrams_df.to_csv(config.OUTPUT_CSV, index=False)
    logging.info(f"Saved top n-grams to {config.OUTPUT_CSV}")

    # Partial text evaluation
    logging.info("Evaluating partial text classification...")
    test_df["first_half"] = test_df["text"].apply(
        lambda x: " ".join(x.split()[:len(x.split())//2])
    )
    
    partial_preds = trainer.predict(Dataset.from_pandas(test_df))
    test_df["partial_pred"] = np.argmax(partial_preds.predictions, axis=1)
    
    # Generate final report
    full_report = classification_report(
        test_df["target"], 
        test_df["predicted_label"],
        target_names=le.classes_,
        digits=4
    )
    
    partial_report = classification_report(
        test_df["target"], 
        test_df["partial_pred"],
        target_names=le.classes_,
        digits=4
    )

    logging.info("\nFull Text Classification Report:\n" + full_report)
    logging.info("\nPartial Text Classification Report:\n" + partial_report)

    # Save final reports
    with open(config.OUTPUT_DIR / "classification_reports.txt", "w") as f:
        f.write("Full Text Classification Report:\n")
        f.write(full_report)
        f.write("\n\nPartial Text Classification Report:\n")
        f.write(partial_report)

    logging.info("Analysis pipeline completed successfully")
def main():
    """Main execution pipeline"""
    config = Config()
    setup_environment(config)
    
    # Data preparation
    df = load_data(config.DATA_DIR)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    train_dataset, val_dataset, test_dataset = prepare_datasets(df, tokenizer)

    # Model initialization and training
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_DIR, 
        num_labels=df["target"].nunique()
    )
    
    trainer = Trainer(
        model=model,
        args=config.TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Perform complete analysis
    perform_analysis(test_dataset, model, tokenizer, config)
 
if __name__ == "__main__":
    main()