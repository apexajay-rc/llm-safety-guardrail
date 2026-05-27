"""
CrisisGuard End-to-End Pipeline

Combines:
- Crisis classification
- Uncertainty-aware routing
- Controlled LLM generation
- Deterministic intervention
- Human escalation pathways

Author: CrisisGuard Research Team
"""

import logging
from typing import Dict, Any

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

from integrations.groq_llm import (
    GroqLLM
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
    End-to-end crisis-aware conversational
    safety pipeline.
    """

    def __init__(
        self,
        groq_api_key: str
    ) -> None:

        logger.info(
            "Initializing CrisisGuard Pipeline..."
        )

        # -------------------------------------------
        # Core Components
        # -------------------------------------------

        self.classifier = CrisisClassifier()

        self.router = SafetyRouter()

        self.intervention_engine = (
            InterventionEngine()
        )

        self.llm = GroqLLM(
            api_key=groq_api_key
        )

        logger.info(
            "Pipeline initialized successfully."
        )

    # ---------------------------------------------------

    def process(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Process user input through
        the full CrisisGuard pipeline.

        Args:
            text:
                User utterance.

        Returns:
            Structured response dictionary.
        """

        logger.info(
            f"Processing input: {text}"
        )

        # -------------------------------------------
        # Step 1: Classification
        # -------------------------------------------

        prediction = self.classifier.predict(
            text
        )

        # -------------------------------------------
        # Step 2: Routing Decision
        # -------------------------------------------

        decision = self.router.route(
            severity=prediction.severity,
            confidence=prediction.confidence
        )

        # -------------------------------------------
        # Step 3A: Intervention Mode
        # -------------------------------------------

        if decision.route == Route.INTERVENTION:

            logger.warning(
                "Intervention route triggered."
            )

            intervention = (
                self.intervention_engine.generate(
                    prediction.severity
                )
            )

            return {

                "input": text,

                "severity":
                    prediction.severity,

                "confidence":
                    prediction.confidence,

                "probabilities":
                    prediction.probabilities,

                "route":
                    decision.route.value,

                "reason":
                    decision.reason,

                "response":
                    intervention.message,

                "escalation":
                    intervention
                    .escalation_recommended
            }

        # -------------------------------------------
        # Step 3B: Human Escalation
        # -------------------------------------------

        if decision.route == Route.ESCALATE:

            logger.warning(
                "Human escalation triggered."
            )

            return {

                "input": text,

                "severity":
                    prediction.severity,

                "confidence":
                    prediction.confidence,

                "probabilities":
                    prediction.probabilities,

                "route":
                    decision.route.value,

                "reason":
                    decision.reason,

                "response":
                    (
                        "This conversation may "
                        "require additional "
                        "human review."
                    ),

                "escalation": True
            }

        # -------------------------------------------
        # Step 3C: Constrained Generation
        # -------------------------------------------

        if decision.route == Route.CONSTRAINED:

            logger.info(
                "Constrained generation route."
            )

            llm_response = self.llm.generate(

                user_input=text,

                constrained=True
            )

            return {

                "input": text,

                "severity":
                    prediction.severity,

                "confidence":
                    prediction.confidence,

                "probabilities":
                    prediction.probabilities,

                "route":
                    decision.route.value,

                "reason":
                    decision.reason,

                "response":
                    llm_response,

                "escalation": False
            }

        # -------------------------------------------
        # Step 3D: Normal Generation
        # -------------------------------------------

        logger.info(
            "Normal generation route."
        )

        llm_response = self.llm.generate(

            user_input=text,

            constrained=False
        )

        return {

            "input": text,

            "severity":
                prediction.severity,

            "confidence":
                prediction.confidence,

            "probabilities":
                prediction.probabilities,

            "route":
                decision.route.value,

            "reason":
                decision.reason,

            "response":
                llm_response,

            "escalation": False
        }


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    API_KEY = "YOUR_GROQ_API_KEY"

    pipeline = CrisisGuardPipeline(
        groq_api_key=API_KEY
    )

    examples = [

        "I had a productive day.",

        "I feel emotionally exhausted.",

        "Nobody would care if I disappeared.",

        "I don't want to live anymore."
    ]

    for text in examples:

        result = pipeline.process(text)

        print("\n================================")
        print(f"INPUT : {result['input']}")

        print(
            f"SEVERITY : "
            f"{result['severity']}"
        )

        print(
            f"CONFIDENCE : "
            f"{result['confidence']:.4f}"
        )

        print(
            f"ROUTE : "
            f"{result['route']}"
        )

        print(
            f"REASON : "
            f"{result['reason']}"
        )

        print(
            f"ESCALATION : "
            f"{result['escalation']}"
        )

        print("\nRESPONSE:")
        print(result["response"])
