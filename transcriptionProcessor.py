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
            return (response.content if hasattr(response, 'content') else f"[ERROR] {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR SIMPLIFY ERROR", str(e)))
            return f"ERROR: Failed to simplify; error: {e}"

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
                    "Task: Translate the following Traditional Chinese into formal and fluent English.\n"
                    "Output: English only. Do NOT include any Chinese text, explanations, or notes.\n\n"
                    f"Text to translate:\n{traditional_text.strip()}"
                )
            )
        )
        try:
            response = LLMInterface.engage(cont)
            return (response.content if hasattr(response, 'content') else f"[ERROR] {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRANSLATE ERROR", str(e)))
            return f"ERROR: Failed to translate into english; error: {e}"

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
            return (response.content if hasattr(response, 'content') else f"[ERROR] {str(response).strip()}")
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR SUMMARY ERROR", str(e)))
            return f"ERROR: Failed to summarise text; error: {e}"

    @staticmethod
    def process(traditional_text: str, tracer: ASTracer) -> tuple:
        try:
            simplified = TranscriptionProcessor.tradToSimp(traditional_text, tracer)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "Simplified Chinese generated.", extraData={"output": simplified}))

            english = TranscriptionProcessor.tradToEng(traditional_text, tracer)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "English translation generated.", extraData={"output": english}))

            summary = TranscriptionProcessor.engSummary(english, tracer)
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "English summary generated.", extraData={"output": summary}))

            return simplified, english, summary

        except Exception as e:
            tracer.addReport(ASReport(
                "TRANSCRIPTIONPROCESSOR PROCESS ERROR",
                f"Failed during processing: {e}"
            ))
            print(f"ERROR: {e}")
            return None, None, None