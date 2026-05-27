"""
CrisisGuard Training Script

Fine-tunes RoBERTa for
crisis severity classification.

Author: apexajay-rc
"""

import logging
import numpy as np
import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

import torch


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

MODEL_NAME = "roberta-base"

NUM_LABELS = 5

MAX_LENGTH = 256

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

def load_dataset():

    """
    Load dataset from CSV.

    Expected format:
    text,label
    """

    logger.info(
        "Loading dataset..."
    )

    train_df = pd.read_csv(
        "data/processed/train.csv"
    )

    val_df = pd.read_csv(
        "data/processed/val.csv"
    )

    train_dataset = Dataset.from_pandas(
        train_df
    )

    val_dataset = Dataset.from_pandas(
        val_df
    )

    return train_dataset, val_dataset


# ---------------------------------------------------
# Tokenization
# ---------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_function(example):

    return tokenizer(

        example["text"],

        truncation=True,

        padding="max_length",

        max_length=MAX_LENGTH
    )


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            labels,

            predictions,

            average="weighted"
        )
    )

    # -----------------------------------------------
    # False Negative Analysis
    # -----------------------------------------------

    fnr = compute_false_negative_rate(
        labels,
        predictions
    )

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "false_negative_rate": fnr
    }


# ---------------------------------------------------
# False Negative Rate
# ---------------------------------------------------

def compute_false_negative_rate(
    labels,
    predictions
):

    """
    Safety-critical metric.

    Measures dangerous misses for
    high-risk crisis labels.
    """

    dangerous_labels = [3, 4]

    total_dangerous = 0

    missed_dangerous = 0

    for true, pred in zip(
        labels,
        predictions
    ):

        if true in dangerous_labels:

            total_dangerous += 1

            if pred < 3:
                missed_dangerous += 1

    if total_dangerous == 0:
        return 0.0

    return (
        missed_dangerous
        / total_dangerous
    )


# ---------------------------------------------------
# Main Training
# ---------------------------------------------------

def train():

    logger.info(
        "Starting training..."
    )

    # -----------------------------------------------
    # Dataset
    # -----------------------------------------------

    train_dataset, val_dataset = (
        load_dataset()
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True
    )

    # -----------------------------------------------
    # Model
    # -----------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(

            MODEL_NAME,

            num_labels=NUM_LABELS
        )
    )

    # -----------------------------------------------
    # Training Arguments
    # -----------------------------------------------

    training_args = TrainingArguments(

        output_dir="./checkpoints",

        evaluation_strategy="epoch",

        save_strategy="epoch",

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=EPOCHS,

        weight_decay=0.01,

        logging_dir="./logs",

        logging_steps=10,

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        save_total_limit=2
    )

    # -----------------------------------------------
    # Trainer
    # -----------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=val_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics
    )

    # -----------------------------------------------
    # Train
    # -----------------------------------------------

    trainer.train()

    # -----------------------------------------------
    # Save Final Model
    # -----------------------------------------------

    trainer.save_model(
        "./models/final_model"
    )

    tokenizer.save_pretrained(
        "./models/final_model"
    )

    logger.info(
        "Training complete."
    )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    train()
