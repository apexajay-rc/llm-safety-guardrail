"""
CrisisGuard Context Manager

Tracks multi-turn conversational context
for evolving crisis-risk estimation.

Author: CrisisGuard Research Team
"""

from dataclasses import dataclass, field
from typing import List, Dict
import statistics
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
# Conversation Turn
# ---------------------------------------------------

@dataclass
class ConversationTurn:
    """
    Represents one conversational step.
    """

    text: str
    severity: int
    confidence: float


# ---------------------------------------------------
# Context State
# ---------------------------------------------------

@dataclass
class ContextState:
    """
    Aggregated conversational state.
    """

    average_severity: float
    max_severity: int
    trend: str
    escalation_risk: bool


# ---------------------------------------------------
# Context Manager
# ---------------------------------------------------

class ContextManager:
    """
    Maintains conversational history
    and computes evolving risk state.
    """

    def __init__(
        self,
        max_history: int = 10
    ) -> None:

        self.max_history = max_history

        self.history: List[
            ConversationTurn
        ] = []

        logger.info(
            f"ContextManager initialized | "
            f"max_history={max_history}"
        )

    # ---------------------------------------------------

    def add_turn(
        self,
        text: str,
        severity: int,
        confidence: float
    ) -> None:
        """
        Add conversational turn.
        """

        turn = ConversationTurn(
            text=text,
            severity=severity,
            confidence=confidence
        )

        self.history.append(turn)

        # Keep fixed window

        if len(self.history) > self.max_history:
            self.history.pop(0)

        logger.info(
            f"Turn added | "
            f"severity={severity}"
        )

    # ---------------------------------------------------

    def compute_state(
        self
    ) -> ContextState:
        """
        Compute conversational risk state.
        """

        if not self.history:

            return ContextState(
                average_severity=0.0,
                max_severity=0,
                trend="STABLE",
                escalation_risk=False
            )

        severities = [
            turn.severity
            for turn in self.history
        ]

        avg = statistics.mean(severities)

        maximum = max(severities)

        # -----------------------------------------------
        # Trend Detection
        # -----------------------------------------------

        trend = "STABLE"

        if len(severities) >= 3:

            recent = severities[-3:]

            if recent[0] < recent[-1]:
                trend = "INCREASING"

            elif recent[0] > recent[-1]:
                trend = "DECREASING"

        # -----------------------------------------------
        # Escalation Risk
        # -----------------------------------------------

        escalation = (
            avg >= 2.5
            or maximum >= 4
            or trend == "INCREASING"
        )

        logger.info(
            f"Context state | "
            f"avg={avg:.2f}, "
            f"max={maximum}, "
            f"trend={trend}"
        )

        return ContextState(
            average_severity=round(avg, 2),
            max_severity=maximum,
            trend=trend,
            escalation_risk=escalation
        )

    # ---------------------------------------------------

    def clear(self) -> None:
        """
        Reset conversation history.
        """

        self.history.clear()

        logger.info(
            "Conversation history cleared."
        )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    manager = ContextManager()

    examples = [

        ("I'm tired lately.", 1, 0.78),

        ("Everything feels exhausting.", 2, 0.82),

        ("Nobody would miss me.", 3, 0.91),

        ("I don't want tomorrow.", 4, 0.95)
    ]

    for text, severity, confidence in examples:

        manager.add_turn(
            text=text,
            severity=severity,
            confidence=confidence
        )

        state = manager.compute_state()

        print("\n================================")
        print(f"INPUT : {text}")

        print(
            f"AVG SEVERITY : "
            f"{state.average_severity}"
        )

        print(
            f"MAX SEVERITY : "
            f"{state.max_severity}"
        )

        print(
            f"TREND : "
            f"{state.trend}"
        )

        print(
            f"ESCALATION RISK : "
            f"{state.escalation_risk}"
        )
