"""
CrisisGuard Groq LLM Integration

Provides controlled LLM generation
with safety-aware prompting.

Author: CrisisGuard Research Team
"""

import logging
from typing import Optional

from groq import Groq


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Groq Client
# ---------------------------------------------------

class GroqLLM:
    """
    Controlled Groq LLM interface.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama3-8b-8192"
    ) -> None:
        """
        Initialize Groq client.
        """

        self.client = Groq(
            api_key=api_key
        )

        self.model_name = model_name

        logger.info(
            f"GroqLLM initialized | "
            f"model={model_name}"
        )

    # ---------------------------------------------------

    def generate(
        self,
        user_input: str,
        constrained: bool = False
    ) -> str:
        """
        Generate response from LLM.

        Args:
            user_input:
                User message.

            constrained:
                Whether to apply
                safety-constrained prompting.

        Returns:
            Generated response.
        """

        logger.info(
            f"Generating response | "
            f"constrained={constrained}"
        )

        # -----------------------------------------------
        # System Prompt
        # -----------------------------------------------

        system_prompt = (
            "You are a calm, supportive, and safe AI assistant. "
            "Do not provide harmful advice, "
            "self-harm encouragement, or manipulative language."
        )

        # -----------------------------------------------
        # Additional constrained behavior
        # -----------------------------------------------

        if constrained:

            system_prompt += (
                " Respond conservatively. "
                "Prioritize emotional safety. "
                "Avoid emotionally intense responses. "
                "Encourage real-world support when appropriate."
            )

        # -----------------------------------------------
        # API Call
        # -----------------------------------------------

        try:

            completion = (
                self.client.chat.completions.create(

                    model=self.model_name,

                    messages=[

                        {
                            "role": "system",
                            "content": system_prompt
                        },

                        {
                            "role": "user",
                            "content": user_input
                        }
                    ],

                    temperature=0.3,
                    max_tokens=256
                )
            )

            response = (
                completion
                .choices[0]
                .message.content
            )

            logger.info(
                "Generation successful."
            )

            return response

        except Exception as e:

            logger.error(
                f"Groq API failure: {str(e)}"
            )

            return (
                "Generation failed safely."
            )


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------

if __name__ == "__main__":

    API_KEY = "YOUR_GROQ_API_KEY"

    llm = GroqLLM(
        api_key=API_KEY
    )

    examples = [

        (
            "I'm emotionally exhausted.",
            True
        ),

        (
            "Tell me a joke.",
            False
        )
    ]

    for text, constrained in examples:

        response = llm.generate(
            user_input=text,
            constrained=constrained
        )

        print("\n================================")
        print(f"INPUT : {text}")

        print(
            f"CONSTRAINED : "
            f"{constrained}"
        )

        print(f"\nRESPONSE:\n{response}")
