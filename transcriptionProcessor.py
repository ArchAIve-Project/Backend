import traceback
from addons import ASTracer, ASReport
from ai import LLMInterface, InteractionContext, Interaction, LMProvider, LMVariant


class TranscriptionProcessor:
    """
    Processes traditional Chinese text by:
    - Converting it to simplified Chinese
    - Translating it to English
    - Summarizing the English version
    """
    
    def extract_content(response):
        # Return the `.content` if it's a message object
        if hasattr(response, "content"):
            return response.content.strip()
        # If it's a string (error), return it as-is
        elif isinstance(response, str):
            return f"[ERROR] {response.strip()}"
        else:
            return "[ERROR] Unknown response type"

    @staticmethod
    def process(traditional_text: str, tracer: ASTracer) -> tuple:
        """
        Takes traditional Chinese text and returns:
        - simplified Chinese
        - English translation
        - summary in English
        """

        try:
            # Convert Traditional → Simplified Chinese
            cont_simp = InteractionContext(
                provider=LMProvider.QWEN,
                variant=LMVariant.QWEN_PLUS
            )
            cont_simp.addInteraction(
                Interaction(
                    role=Interaction.Role.USER,
                    content=f"Convert the following traditional Chinese text to simplified Chinese. Only output simplified Chinese:\n\n{traditional_text.strip()}"
                )
            )
            # simp_response = LLMInterface.engage(cont_simp)
            # print(simp_response)
            # simplified = simp_response.content.strip()
            simplified = TranscriptionProcessor.extract_content(LLMInterface.engage(cont_simp))
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "Simplified Chinese generated."))

            # Translate Traditional Chinese → English
            cont_en = InteractionContext(
                provider=LMProvider.QWEN,
                variant=LMVariant.QWEN_PLUS
            )
            cont_en.addInteraction(
                Interaction(
                    role=Interaction.Role.USER,
                    content=f"Translate the following traditional Chinese text into English. Only output English:\n\n{traditional_text.strip()}"
                )
            )
            # en_response = LLMInterface.engage(cont_en)
            # print(en_response)
            # english = en_response.content.strip()
            english = TranscriptionProcessor.extract_content(LLMInterface.engage(cont_en))
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "English translation generated."))

            # Summarize the English Text
            cont_sum = InteractionContext(
                provider=LMProvider.OPENAI,
                variant=LMVariant.GPT_4O_MINI
            )
            cont_sum.addInteraction(
                Interaction(
                    role=Interaction.Role.USER,
                    content=f"Summarize the following English text concisely:\n\n{english}"
                )
            )
            # sum_response = LLMInterface.engage(cont_sum)
            # print(sum_response)
            # summary = sum_response.content.strip()
            summary = TranscriptionProcessor.extract_content(LLMInterface.engage(cont_sum))
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS", "English summary generated."))
            
            tracer.end()

            return simplified, english, summary


        except Exception as e:
            tracer.addReport(ASReport("TRANSCRIPTIONPROCESSOR PROCESS ERROR", f"Failed during processing: {e}", {
                "traceback": traceback.format_exc()
            }))
            print("ERROR: {}".format(e))
            tracer.end()
            
            return "", "", ""

