import traceback
from addons import ASTracer, ASReport
from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant, Tool

class TranscriptionProcessor:
    """
    Handles the processing of traditional Chinese text by:
    - Converting to simplified Chinese
    - Translating to English
    - Summarizing the translation
    """

    @staticmethod
    def tradToSimp(traditional_text: str, tracer: ASTracer) -> str:
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_PLUS
        )
        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=f"Convert the following traditional Chinese text to simplified Chinese. Only output simplified Chinese:\n\n{traditional_text.strip()}"
            )
        )
        try:
            response = LLMInterface.engage(cont)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOSIMP", "Simplified Chinese generated."))
            return (response.content if hasattr(response, 'content') else f"ERROR {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOSIMP ERROR", str(e)))
            return f"ERROR: Failed to simplify: {e}"

    @staticmethod
    def tradToEng(traditional_text: str, tracer: ASTracer) -> str:
        cont = InteractionContext(
            provider=LMProvider.QWEN,
            variant=LMVariant.QWEN_PLUS
        )
        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=(
                    "Translate the following Traditional Chinese into formal and fluent English.\n"
                    "Output: English only. Do NOT include any Chinese text, explanations, or notes.\n\n"
                    f"Text to translate:\n\n{traditional_text.strip()}"
                )
            )
        )
        try:
            response = LLMInterface.engage(cont)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOENG", "English translation generated."))
            return (response.content if hasattr(response, 'content') else f"ERROR {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOENG ERROR", str(e)))
            return f"ERROR: Failed to translate into english: {e}"

    @staticmethod
    def engSummary(english_text: str, tracer: ASTracer) -> str:
        cont = InteractionContext(
            provider=LMProvider.OPENAI,
            variant=LMVariant.GPT_4O_MINI
        )
        cont.addInteraction(
            Interaction(
                role=Interaction.Role.USER,
                content=f"Summarize the following English text concisely:\n\n{english_text.strip()}"
            )
        )
        try:
            response = LLMInterface.engage(cont)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR ENGSUMMARY", "English summary generated."))
            return (response.content if hasattr(response, 'content') else f"ERROR {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR ENGSUMMARY ERROR", str(e)))
            return f"ERROR: Failed to summarise text: {e}"

    @staticmethod
    def process(traditional_text: str, tracer: ASTracer) -> tuple:
        try:
            simplified = TranscriptionProcessor.tradToSimp(traditional_text, tracer)

            english = TranscriptionProcessor.tradToEng(traditional_text, tracer)

            summary = TranscriptionProcessor.engSummary(english, tracer)

            return simplified, english, summary

        except Exception as e:
            tracer.addReport(ASReport(
                "TRANSCRIPTIONPROCESSOR PROCESS ERROR",
                f"Failed during processing: {e}"
            ))
            print(f"ERROR: {e}")
            return "ERROR: {e}"