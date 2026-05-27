"""
CrisisGuard Safety Router

This module defines routing policies for crisis-aware conversational AI.
It determines whether user input should be:

1. Sent to the LLM normally
2. Sent through constrained generation
3. Routed to intervention mode
4. Escalated for human review

Author: CrisisGuard Research Team
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
import logging


# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Routing Modes
# ---------------------------------------------------

class Route(Enum):
    """
    Available routing decisions.
    """

    NORMAL = "normal_generation"
    CONSTRAINED = "constrained_generation"
    INTERVENTION = "intervention_mode"
    ESCALATE = "human_escalation"


# ---------------------------------------------------
# Routing Result
# ---------------------------------------------------

@dataclass
class RoutingDecision:
    """
    Structured routing output.
    """

    severity: int
    confidence: float
    route: Route
    reason: str


# ---------------------------------------------------
# Safety Router
# ---------------------------------------------------

class SafetyRouter:
    """
    Crisis-aware routing engine.

    Determines how the conversational system should
    respond based on predicted crisis severity and
    model confidence.
    """

    def __init__(
        self,
        intervention_threshold: int = 3,
        confidence_threshold: float = 0.65
    ) -> None:
        """
        Initialize routing thresholds.

        Args:
            intervention_threshold:
                Severity level triggering intervention.

            confidence_threshold:
                Minimum confidence required before
                trusting automated decision-making.
        """

        self.intervention_threshold = intervention_threshold
        self.confidence_threshold = confidence_threshold

        logger.info(
            "SafetyRouter initialized | "
            f"intervention_threshold={intervention_threshold}, "
            f"confidence_threshold={confidence_threshold}"
        )

    # ---------------------------------------------------

    def route(
        self,
        severity: int,
        confidence: float
    ) -> RoutingDecision:
        """
        Determine routing policy.

        Args:
            severity:
                Crisis severity level (0-4)

            confidence:
                Model confidence score (0-1)

        Returns:
            RoutingDecision
        """

        logger.info(
            f"Received prediction | "
            f"severity={severity}, confidence={confidence:.3f}"
        )

        # -----------------------------------------------
        # Human escalation for uncertainty
        # -----------------------------------------------

        if confidence < self.confidence_threshold:
            logger.warning(
                "Low confidence detected. Escalating."
            )

            return RoutingDecision(
                severity=severity,
                confidence=confidence,
                route=Route.ESCALATE,
                reason="Low confidence prediction"
            )

        # -----------------------------------------------
        # Intervention mode
        # -----------------------------------------------

        if severity >= self.intervention_threshold:

            logger.warning(
                "High-risk crisis severity detected."
            )

            return RoutingDecision(
                severity=severity,
                confidence=confidence,
                route=Route.INTERVENTION,
                reason="High-risk crisis detected"
            )

        # -----------------------------------------------
        # Constrained generation
        # -----------------------------------------------

        if severity == 2:

            logger.info(
                "Moderate distress detected."
            )

            return RoutingDecision(
                severity=severity,
                confidence=confidence,
                route=Route.CONSTRAINED,
                reason="Moderate emotional distress"
            )

        # -----------------------------------------------
        # Normal generation
        # -----------------------------------------------

        logger.info(
            "Low-risk interaction."
        )

        return RoutingDecision(
            severity=severity,
            confidence=confidence,
            route=Route.NORMAL,
            reason="Low-risk interaction"
        )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    router = SafetyRouter()

    examples = [
        {"severity": 0, "confidence": 0.94},
        {"severity": 2, "confidence": 0.88},
        {"severity": 4, "confidence": 0.91},
        {"severity": 3, "confidence": 0.42},
    ]

    for item in examples:

        decision = router.route(
            severity=item["severity"],
            confidence=item["confidence"]
        )

        print("\n--------------------------------")
        print(f"Severity : {decision.severity}")
        print(f"Confidence : {decision.confidence:.2f}")
        print(f"Route : {decision.route.value}")
        print(f"Reason : {decision.reason}")
