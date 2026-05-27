"""
CrisisGuard Uncertainty Estimation

Implements uncertainty analysis utilities
for trustworthy crisis classification.

Author: CrisisGuard Research Team
"""

from dataclasses import dataclass
from typing import Dict
import math
import logging


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Uncertainty Result
# ---------------------------------------------------

@dataclass
class UncertaintyResult:
    """
    Structured uncertainty metrics.
    """

    entropy: float
    max_probability: float
    uncertainty_level: str


# ---------------------------------------------------
# Uncertainty Estimator
# ---------------------------------------------------

class UncertaintyEstimator:
    """
    Computes uncertainty metrics from
    probability distributions.
    """

    def __init__(self) -> None:

        logger.info(
            "UncertaintyEstimator initialized."
        )

    # ---------------------------------------------------

    def compute_entropy(
        self,
        probabilities: Dict[int, float]
    ) -> float:
        """
        Compute entropy of probability distribution.

        Higher entropy = more uncertainty.
        """

        entropy = 0.0

        for p in probabilities.values():

            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    # ---------------------------------------------------

    def classify_uncertainty(
        self,
        entropy: float
    ) -> str:
        """
        Categorize uncertainty level.
        """

        if entropy < 0.5:
            return "LOW"

        if entropy < 1.0:
            return "MODERATE"

        return "HIGH"

    # ---------------------------------------------------

    def analyze(
        self,
        probabilities: Dict[int, float]
    ) -> UncertaintyResult:
        """
        Analyze uncertainty metrics.
        """

        entropy = self.compute_entropy(
            probabilities
        )

        max_probability = max(
            probabilities.values()
        )

        uncertainty_level = (
            self.classify_uncertainty(
                entropy
            )
        )

        logger.info(
            f"Entropy={entropy:.4f} | "
            f"MaxProb={max_probability:.4f} | "
            f"Uncertainty={uncertainty_level}"
        )

        return UncertaintyResult(
            entropy=round(entropy, 4),
            max_probability=round(
                max_probability,
                4
            ),
            uncertainty_level=uncertainty_level
        )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    estimator = UncertaintyEstimator()

    examples = [

        # -------------------------------------------
        # Very confident
        # -------------------------------------------

        {
            0: 0.97,
            1: 0.01,
            2: 0.01,
            3: 0.005,
            4: 0.005
        },

        # -------------------------------------------
        # Ambiguous
        # -------------------------------------------

        {
            0: 0.22,
            1: 0.18,
            2: 0.20,
            3: 0.19,
            4: 0.21
        },

        # -------------------------------------------
        # Moderate uncertainty
        # -------------------------------------------

        {
            0: 0.10,
            1: 0.15,
            2: 0.55,
            3: 0.10,
            4: 0.10
        }
    ]

    for probs in examples:

        result = estimator.analyze(probs)

        print("\n================================")
        print(f"Entropy : {result.entropy}")
        print(f"Max Probability : "
              f"{result.max_probability}")
        print(f"Uncertainty : "
              f"{result.uncertainty_level}")
