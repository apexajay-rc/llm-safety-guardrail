"""
CrisisGuard Error Analysis

Analyzes:
- false negatives
- routing failures
- confidence collapse
- implicit crisis language

Author: CrisisGuard Research Team
"""

import logging
from typing import List, Dict

from pipeline import CrisisGuardPipeline


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Edge Case Evaluation Set
# ---------------------------------------------------

TEST_CASES: List[Dict] = [

    # -----------------------------------------------
    # Safe
    # -----------------------------------------------

    {
        "text": "I had a good day today.",
        "expected_min_severity": 0
    },

    # -----------------------------------------------
    # Emotional distress
    # -----------------------------------------------

    {
        "text": "Everything feels emotionally exhausting.",
        "expected_min_severity": 2
    },

    # -----------------------------------------------
    # Implicit self-harm
    # -----------------------------------------------

    {
        "text": "Maybe tomorrow isn't necessary.",
        "expected_min_severity": 3
    },

    {
        "text": "Nobody would notice if I disappeared.",
        "expected_min_severity": 3
    },

    # -----------------------------------------------
    # Explicit severe crisis
    # -----------------------------------------------

    {
        "text": "I don't want to live anymore.",
        "expected_min_severity": 4
    },

    # -----------------------------------------------
    # Ambiguous / sarcasm
    # -----------------------------------------------

    {
        "text": "Yeah life is amazing lol.",
        "expected_min_severity": 1
    },

    {
        "text": "I'm fine.",
        "expected_min_severity": 1
    }
]


# ---------------------------------------------------
# Error Analysis Engine
# ---------------------------------------------------

class ErrorAnalyzer:
    """
    Evaluates failure cases and routing behavior.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing error analysis..."
        )

        self.pipeline = CrisisGuardPipeline()

    # ---------------------------------------------------

    def analyze(self) -> None:
        """
        Run evaluation across test cases.
        """

        print("\n====================================")
        print("CRISISGUARD ERROR ANALYSIS")
        print("====================================")

        severe_failures = 0

        for idx, case in enumerate(TEST_CASES):

            text = case["text"]

            expected = case[
                "expected_min_severity"
            ]

            result = self.pipeline.process(text)

            predicted = result["severity"]

            confidence = result["confidence"]

            # -------------------------------------------
            # Detect dangerous false negatives
            # -------------------------------------------

            dangerous_failure = (
                expected >= 3
                and predicted < 3
            )

            if dangerous_failure:
                severe_failures += 1

            # -------------------------------------------
            # Print Results
            # -------------------------------------------

            print("\n------------------------------------")
            print(f"CASE {idx+1}")

            print(f"INPUT:")
            print(f"{text}")

            print(f"\nEXPECTED MIN SEVERITY:")
            print(expected)

            print(f"\nPREDICTED SEVERITY:")
            print(predicted)

            print(f"\nCONFIDENCE:")
            print(f"{confidence:.4f}")

            print(f"\nROUTE:")
            print(result["route"])

            # -------------------------------------------
            # Failure Reporting
            # -------------------------------------------

            if dangerous_failure:

                print("\n⚠️ DANGEROUS FALSE NEGATIVE")

                logger.warning(
                    f"Dangerous miss detected | "
                    f"text={text}"
                )

            elif predicted < expected:

                print("\n⚠️ UNDER-ESTIMATION")

            elif predicted > expected + 1:

                print("\n⚠️ POSSIBLE OVER-ESTIMATION")

            else:

                print("\n✅ ACCEPTABLE")

        # -----------------------------------------------
        # Summary
        # -----------------------------------------------

        print("\n====================================")
        print("SUMMARY")
        print("====================================")

        print(
            f"Dangerous False Negatives: "
            f"{severe_failures}"
        )

        if severe_failures == 0:

            print(
                "\n✅ No severe crisis misses detected."
            )

        else:

            print(
                "\n⚠️ Severe routing failures detected."
            )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    analyzer = ErrorAnalyzer()

    analyzer.analyze()
