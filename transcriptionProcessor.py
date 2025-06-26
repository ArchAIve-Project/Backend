import traceback
from addons import ASTracer, ASReport
from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant, Tool

class TranscriptionProcessor:
    """
    Processes traditional Chinese text through conversion, translation, and summarization.

    Provides static methods to convert traditional Chinese to simplified Chinese,
    translate traditional Chinese to English, and summarize the English translation.

    ## Usage:
    ```python
    simplified, english, summary = TranscriptionProcessor.process(traditional_text, tracer)
    ```
    """


    @staticmethod
    def tradToSimp(traditional_text: str, tracer: ASTracer) -> str:
        """
        Converts traditional Chinese text to simplified Chinese using an LLM.

        Sends the input traditional Chinese text to a language model for conversion,
        and returns the simplified Chinese output. Logs success or failure to the tracer.

        Args:
            traditional_text (str): Text in traditional Chinese characters.
            tracer (ASTracer): Tracer object for logging the conversion process.

        Returns:
            str: Simplified Chinese text if successful, or error message on failure.

        ## Usage:
        ```python
        simplified_text = CCRPipeline.tradToSimp(traditional_text, tracer)
        ```
        """

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
            if isinstance(response, str):
                raise Exception("ERROR: Failed to carry out LLM translation from Traditional Chinese to Simplified Chinese; response: {}".format(response))

            content = response.content if hasattr(response, 'content') else f"ERROR {str(response).strip()}"
            charCount = len(content.strip())
            
            tracer.addReport(ASReport(
                "TRANSCRIPTIONPROCESSOR TRADTOSIMP",
                f"Simplified Chinese generated. Number of characters: {charCount}"
            ))

            return content

        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOSIMP ERROR", str(e)))
            return f"ERROR: Failed to simplify: {e}"

    @staticmethod
    def tradToEng(traditional_text: str, tracer: ASTracer) -> str:
        """
        Translates traditional Chinese text into formal and fluent English using an LLM.

        Sends the traditional Chinese input to a language model that outputs English only,
        without including any Chinese text or explanations. Logs translation status to the tracer.

        Args:
            traditional_text (str): Text in traditional Chinese characters.
            tracer (ASTracer): Tracer object for logging the translation process.

        Returns:
            str: English translation if successful, or error message on failure.

        ## Usage:
        ```python
        english_text = CCRPipeline.tradToEng(traditional_text, tracer)
        ```
        """

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
            if isinstance(response, str):
                raise Exception("ERROR: Failed to carry out LLM translation from Traditional Chinese to English; response: {}".format(response))
            
            content = response.content if hasattr(response, 'content') else f"ERROR {str(response).strip()}"
            charCount = len(content.strip())
            
            tracer.addReport(ASReport(
                "TRANSCRIPTIONPROCESSOR TRADTOENG", 
                f"English translation generated. Number of characters: {charCount}"
            ))
            
            return content
        
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR TRADTOENG ERROR", str(e)))
            return f"ERROR: Failed to translate into english: {e}"

    @staticmethod
    def engSummary(english_text: str, tracer: ASTracer) -> str:
        """
        Generates a concise summary of English text using an LLM.

        Sends the input English text to a language model to produce a brief summary,
        and logs the result to the tracer.

        Args:
            english_text (str): The English text to summarize.
            tracer (ASTracer): Tracer object for logging the summarization process.

        Returns:
            str: Concise summary of the input text, or error message on failure.

        ## Usage:
        ```python
        summary = CCRPipeline.engSummary(english_text, tracer)
        ```
        """

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
            if isinstance(response, str):
                raise Exception("ERROR: Failed to carry out LLM summarization in English; response: {}".format(response))
            
            content = response.content
            charCount = len(content.strip())
            
            tracer.addReport(ASReport(
                "TRANSCRIPTIONPROCESSOR ENGSUMMARY", 
                f"English summary generated. Number of characters: {charCount}"
            ))
            
            return content
        
        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR ENGSUMMARY ERROR", str(e)))
            return f"ERROR: Failed to summarise text: {e}"

    @staticmethod
    def process(traditional_text: str, tracer: ASTracer) -> tuple:
        """
        Processes traditional Chinese text through conversion, translation, and summarization.

        Performs the following sequential steps:
        1. Converts traditional Chinese to simplified Chinese.
        2. Translates traditional Chinese to English.
        3. Summarizes the English translation.

        Args:
            traditional_text (str): Input text in traditional Chinese.
            tracer (ASTracer): Tracer object for logging each step.

        Returns:
            tuple: A tuple containing (simplified Chinese, English translation, English summary),
                   or an error message string if processing fails.

        ## Usage:
        ```python
        simp, eng, summ = TranscriptionProcessor.process(traditional_text, tracer)
        ```
        """
        
        simplified = TranscriptionProcessor.tradToSimp(traditional_text, tracer)

        english = TranscriptionProcessor.tradToEng(traditional_text, tracer)
        if english.startswith("ERROR:"):
            return simplified, english, None

        summary = TranscriptionProcessor.engSummary(english, tracer)
        return simplified, english, summary