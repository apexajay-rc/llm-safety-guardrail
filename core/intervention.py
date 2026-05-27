"""
CrisisGuard Intervention Engine

Defines safe intervention responses for
high-risk conversational situations.

Author: CrisisGuard Research Team
"""

from dataclasses import dataclass
from typing import Dict
import random
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
# Intervention Response Structure
# ---------------------------------------------------

@dataclass
class InterventionResponse:
    """
    Structured intervention response.
    """

    severity: int
    message: str
    escalation_recommended: bool


# ---------------------------------------------------
# Crisis Intervention Engine
# ---------------------------------------------------

class InterventionEngine:
    """
    Generates safe intervention responses
    based on crisis severity.
    """

    def __init__(self) -> None:

        self.responses: Dict[int, list] = {

            # ---------------------------------------
            # Level 2 — Emotional Distress
            # ---------------------------------------

            2: [
                (
                    "I'm sorry you're going through something difficult. "
                    "You do not have to handle everything alone."
                ),

                (
                    "It sounds like things may feel emotionally heavy right now. "
                    "Reaching out to someone you trust could help."
                )
            ],

            # ---------------------------------------
            # Level 3 — Self-Harm Risk
            # ---------------------------------------

            3: [
                (
                    "I'm really concerned about what you're expressing. "
                    "Please consider contacting someone you trust or a mental health professional."
                ),

                (
                    "You deserve support and safety right now. "
                    "If possible, reach out to a trusted person nearby."
                )
            ],

            # ---------------------------------------
            # Level 4 — Severe Crisis
            # ---------------------------------------

            4: [
                (
                    "Your safety matters. "
                    "Please contact emergency services, a crisis hotline, "
                    "or someone you trust immediately."
                ),

                (
                    "I'm deeply concerned for your immediate safety. "
                    "Please seek urgent support from a trusted person or professional."
                )
            ]
        }

        logger.info("InterventionEngine initialized.")

    # ---------------------------------------------------

    def generate(
        self,
        severity: int
    ) -> InterventionResponse:
        """
        Generate safe intervention response.

        Args:
            severity:
                Crisis severity level.

        Returns:
            InterventionResponse
        """

        logger.info(
            f"Generating intervention response "
            f"for severity={severity}"
        )

        # -----------------------------------------------
        # Default safe fallback
        # -----------------------------------------------

        if severity not in self.responses:

            logger.warning(
                "Unknown severity level received."
            )

            return InterventionResponse(
                severity=severity,
                message=(
                    "I'm here to support safe conversation. "
                    "Please consider reaching out to someone you trust."
                ),
                escalation_recommended=False
            )

        # -----------------------------------------------
        # Select randomized safe response
        # -----------------------------------------------

        selected = random.choice(
            self.responses[severity]
        )

        escalation = severity >= 3

        return InterventionResponse(
            severity=severity,
            message=selected,
            escalation_recommended=escalation
        )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    engine = InterventionEngine()

    test_levels = [2, 3, 4]

    for level in test_levels:

        response = engine.generate(level)

        print("\n--------------------------------")
        print(f"Severity : {response.severity}")
        print(f"Escalation : {response.escalation_recommended}")
        print(f"Message : {response.message}")
