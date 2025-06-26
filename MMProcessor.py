from ccrPipeline import CCRPipeline
from transcriptionProcessor import TranscriptionProcessor
from addons import ArchSmith, ASTracer
from NERPipeline import NERPipeline
import os

class MMProcessor:
    '''
    MMProcessor orchestrates the full multi-modal OCR + NLP pipeline.

    Components:
    - ccrProcess(): Performs character recognition and optional correction using LLM.
    - transcriptionProcess(): Converts traditional Chinese to simplified Chinese, translates to English, and summarizes.
    - mmProcess(): Full sequence combining recognition and transcription.
    '''

    @staticmethod
    def ccrProcess(image_path: str, gt_path: str, tracer, useLLMCorrection=True) -> dict:
        """
        Runs OCR recognition on the image and returns a result dictionary.
        """
        result = CCRPipeline.transcribe(
            image_path=image_path,
            tracer=tracer,
            computeAccuracy=True,
            useLLMCorrection=useLLMCorrection,
            gtPath=gt_path
        )
        return result

    @staticmethod
    def transcriptionProcess(traditional_text: str, tracer) -> tuple:
        """
        Processes the recognized traditional Chinese text:
        - Converts to Simplified Chinese
        - Translates to English
        - Summarizes
        """
        return TranscriptionProcessor.process(traditional_text, tracer)

    @staticmethod
    def nerProcess(enTranslation: str, tracer):
        """
        
        """
        return NERPipeline.predict(enTranslation, tracer)

    @staticmethod
    def mmProcess(image_path: str, gt_path: str, tracer: ASTracer):
        """
        Executes the complete multi-modal pipeline and returns the results.
        """

        result = MMProcessor.ccrProcess(image_path, gt_path, tracer)

        if isinstance(result, dict) and result.get("transcription", "").lower().startswith("error"):
            tracer.end()
            ArchSmith.persist()
            return {
                "error": result["transcription"]
            }

        simCN, eng, summary = MMProcessor.transcriptionProcess(result["transcription"], tracer)
        labels = MMProcessor.nerProcess(eng, tracer)

        tracer.end()
        ArchSmith.persist()

        return {
            "image": os.path.basename(image_path),
            "traditional_chinese": result["transcription"],
            "correction_applied": result["corrected"],
            "pre_correction_accuracy": result.get("preCorrectionAccuracy"),
            "post_correction_accuracy": result.get("postCorrectionAccuracy"),
            "simplified_chinese": simCN,
            "english_translation": eng,
            "summary": summary,
            "ner_labels": labels
        }