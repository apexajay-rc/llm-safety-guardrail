"""
CrisisGuard RoBERTa Classifier

Transformer-based crisis severity classifier.

Author: CrisisGuard Research Team
"""

from dataclasses import dataclass
from typing import Dict, Any

import logging
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Prediction Output
# ---------------------------------------------------

@dataclass
class PredictionResult:
    """
    Structured prediction output.
    """

    severity: int
    confidence: float
    probabilities: Dict[int, float]


# ---------------------------------------------------
# Crisis Classifier
# ---------------------------------------------------

class CrisisClassifier:
    """
    Transformer-based crisis classifier.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 5
    ) -> None:
        """
        Initialize model and tokenizer.
        """

        logger.info(
            f"Loading model: {model_name}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                num_labels=num_labels
            )
        )

        self.model.eval()

        logger.info("Model initialized successfully.")

    # ---------------------------------------------------

    def predict(
        self,
        text: str
    ) -> PredictionResult:
        """
        Predict crisis severity.

        Args:
            text:
                Input user utterance.

        Returns:
            PredictionResult
        """

        logger.info(
            f"Running inference on text: {text}"
        )

        # -----------------------------------------------
        # Tokenization
        # -----------------------------------------------

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        # -----------------------------------------------
        # Forward Pass
        # -----------------------------------------------

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits

            probabilities = F.softmax(
                logits,
                dim=-1
            )[0]

        # -----------------------------------------------
        # Prediction
        # -----------------------------------------------

        severity = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            torch.max(probabilities).item()
        )

        probability_dict = {
            i: round(float(probabilities[i]), 4)
            for i in range(len(probabilities))
        }

        logger.info(
            f"Prediction complete | "
            f"severity={severity}, "
            f"confidence={confidence:.4f}"
        )

        return PredictionResult(
            severity=severity,
            confidence=confidence,
            probabilities=probability_dict
        )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    classifier = CrisisClassifier()

    examples = [

        "I had a good day today.",

        "I feel emotionally exhausted.",

        "Nobody would care if I disappeared.",

        "I do not want to live anymore."
    ]

    for text in examples:

        result = classifier.predict(text)

        print("\n--------------------------------")
        print(f"Input : {text}")
        print(f"Severity : {result.severity}")
        print(f"Confidence : {result.confidence:.4f}")
        print(f"Probabilities : {result.probabilities}")
