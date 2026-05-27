"""
CrisisGuard End-to-End Pipeline

Combines:
- Crisis classification
- Safety routing
- Intervention handling

Author: CrisisGuard Research Team
"""

import logging

from models.roberta_classifier import (
    CrisisClassifier
)

from core.safety_router import (
    SafetyRouter,
    Route
)

from core.intervention import (
    InterventionEngine
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
# CrisisGuard Pipeline
# ---------------------------------------------------

class CrisisGuardPipeline:
    """
    End-to-end crisis-aware AI safety pipeline.
    """

    def __init__(self) -> None:

        logger.info(
            "Initializing CrisisGuard pipeline..."
        )

        self.classifier = CrisisClassifier()

        self.router = SafetyRouter()

        self.intervention_engine = (
            InterventionEngine()
        )

        logger.info(
            "Pipeline initialized successfully."
        )

    # ---------------------------------------------------

    def process(
        self,
        text: str
    ) -> dict:
        """
        Process user input through full pipeline.

        Args:
            text:
                User utterance.

        Returns:
            Structured pipeline response.
        """

        logger.info(
            f"Processing input: {text}"
        )

        # -----------------------------------------------
        # Step 1: Crisis Classification
        # -----------------------------------------------

        prediction = self.classifier.predict(text)

        # -----------------------------------------------
        # Step 2: Safety Routing
        # -----------------------------------------------

        decision = self.router.route(
            severity=prediction.severity,
            confidence=prediction.confidence
        )

        # -----------------------------------------------
        # Step 3: Response Generation
        # -----------------------------------------------

        if decision.route == Route.INTERVENTION:

            intervention = (
                self.intervention_engine.generate(
                    prediction.severity
                )
            )

            return {

                "input": text,

                "severity": prediction.severity,

                "confidence": prediction.confidence,

                "route": decision.route.value,

                "reason": decision.reason,

                "response": intervention.message,

                "escalation":
                    intervention.escalation_recommended
            }

        # -----------------------------------------------
        # Human Escalation
        # -----------------------------------------------

        if decision.route == Route.ESCALATE:

            return {

                "input": text,

                "severity": prediction.severity,

                "confidence": prediction.confidence,

                "route": decision.route.value,

                "reason": decision.reason,

                "response":
                    "This conversation may require "
                    "additional human review.",

                "escalation": True
            }

        # -----------------------------------------------
        # Constrained Generation
        # -----------------------------------------------

        if decision.route == Route.CONSTRAINED:

            return {

                "input": text,

                "severity": prediction.severity,

                "confidence": prediction.confidence,

                "route": decision.route.value,

                "reason": decision.reason,

                "response":
                    "Constrained supportive generation "
                    "would occur here.",

                "escalation": False
            }

        # -----------------------------------------------
        # Normal Generation
        # -----------------------------------------------

        return {

            "input": text,

            "severity": prediction.severity,

            "confidence": prediction.confidence,

            "route": decision.route.value,

            "reason": decision.reason,

            "response":
                "Normal LLM generation allowed.",

            "escalation": False
        }


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    pipeline = CrisisGuardPipeline()

    examples = [

        "I had a productive day.",

        "I feel emotionally exhausted lately.",

        "Nobody would care if I disappeared.",

        "I don't want to live anymore."
    ]

    for text in examples:

        result = pipeline.process(text)

        print("\n================================")
        print(f"INPUT : {result['input']}")
        print(f"SEVERITY : {result['severity']}")
        print(f"CONFIDENCE : {result['confidence']:.4f}")
        print(f"ROUTE : {result['route']}")
        print(f"REASON : {result['reason']}")
        print(f"ESCALATION : {result['escalation']}")
        print(f"RESPONSE : {result['response']}")
