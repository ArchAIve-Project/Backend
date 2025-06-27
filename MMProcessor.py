from ccrPipeline import CCRPipeline
from transcriptionProcessor import TranscriptionProcessor
from addons import ArchSmith, ASTracer, ASReport
from NERPipeline import NERPipeline
import os

class MMProcessor:
    '''
    MMProcessor orchestrates the full multi-modal OCR + NLP pipeline.

    This class provides static methods that:
    - Perform Optical Character Recognition (OCR) on traditional Chinese text.
    - Optionally apply Large Language Model (LLM) correction to OCR output.
    - Convert traditional Chinese text to simplified Chinese.
    - Translate the simplified Chinese to English.
    - Generate an English summary of the translation.
    - Extract named entities using a fine-tuned BERT NER model.

    Components:
    - ccrProcess(): OCR and optional LLM correction.
    - transcriptionProcess(): Chinese-to-English translation pipeline.
    - nerProcess(): Named Entity Recognition on English translation.
    - mmProcess(): Full pipeline execution.
    '''

    @staticmethod
    def ccrProcess(image_path: str, tracer: ASTracer, useLLMCorrection=True, gt_path: str = None) -> dict:
        """
        Runs OCR recognition on the image and returns a result dictionary.

        Args:
            image_path (str): Path to the image file.
            tracer (ASTracer): Tracer object for logging and reporting.
            useLLMCorrection (bool): If True, applies LLM correction to OCR output.
            gt_path (str, optional): Ground truth .txt file path for accuracy computation.

        Returns:
            dict: Dictionary containing OCR transcription, correction flag, and optional accuracy scores.
        """
        try:
            result = CCRPipeline.transcribe(
                image_path=image_path,
                tracer=tracer,
                computeAccuracy=bool(gt_path),
                useLLMCorrection=useLLMCorrection,
                gtPath=gt_path
            )
            if isinstance(result, str) and result.startswith("ERROR"):
                tracer.addReport(ASReport("MMPROCESSOR CCRPROCESS ERROR", result))
                return {"transcription": result}

            tracer.addReport(ASReport(
                "MMPROCESSOR CCRPROCESS",
                f"Processed image {image_path}; gtPath={gt_path}; LLMCorrection={useLLMCorrection}"
            ))
            return result

        except Exception as e:
            msg = f"Exception during CCR process: {e}"
            tracer.addReport(ASReport("MMPROCESSOR CCRPROCESS EXCEPTION", msg))
            return {"transcription": f"ERROR: {msg}"}

    @staticmethod
    def transcriptionProcess(traditional_text: str, tracer: ASTracer) -> tuple:
        """
        Converts and processes traditional Chinese text.

        Steps:
        - Converts to Simplified Chinese.
        - Translates to English.
        - Generates a summary of the English translation.

        Args:
            traditional_text (str): OCR output in traditional Chinese.
            tracer (ASTracer): Tracer object for logging and reporting.

        Returns:
            tuple: (simplified_chinese, english_translation, summary)
        """
        try:
            result = TranscriptionProcessor.process(traditional_text, tracer)

            if isinstance(result, tuple):
                simplified, english, summary = result
                if isinstance(english, str) and english.startswith("ERROR:"):
                    tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTION ERROR", english))
                return simplified, english, summary

            tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTION ERROR", "Unexpected result structure"))
            return traditional_text, "ERROR: Translation failed", None

        except Exception as e:
            msg = f"Exception during transcription: {e}"
            tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTION EXCEPTION", msg))
            return traditional_text, f"ERROR: {msg}", None


    @staticmethod
    def nerProcess(enTranslation: str, tracer: ASTracer):
        """
        Applies Named Entity Recognition (NER) on the English translation.

        Args:
            enTranslation (str): Translated English text.
            tracer (ASTracer): Tracer object for logging and reporting.

        Returns:
            list: List of named entity tuples (token, label).
        """
        try:
            result = NERPipeline.predict(enTranslation, tracer)
            if isinstance(result, str) and result.startswith("ERROR"):
                tracer.addReport(ASReport("MMPROCESSOR NER ERROR", result))
                return []
            return result

        except Exception as e:
            msg = f"Exception during NER prediction: {e}"
            tracer.addReport(ASReport("MMPROCESSOR NER EXCEPTION", msg))
            return []

    @staticmethod
    def mmProcess(image_path: str, tracer: ASTracer, gt_path: str = None):
        """
        Executes the complete multi-modal pipeline:
        - Performs OCR and optional correction.
        - Converts and translates the recognized text.
        - Extracts named entities from the English output.

        Args:
            image_path (str): Path to the input image.
            tracer (ASTracer): Tracer object for pipeline tracking.
            gt_path (str, optional): Optional ground truth path for computing accuracy.

        Returns:
            dict: Final pipeline output with all intermediate results.
        """
        try:
            result = MMProcessor.ccrProcess(image_path=image_path, tracer=tracer, gt_path=gt_path)

            if not isinstance(result, dict) or result.get("transcription", "").lower().startswith("error"):
                tracer.end()
                ArchSmith.persist()
                return {"error": result.get("transcription", "Unknown error in CCR process.")}

            simCN, eng, summary = MMProcessor.transcriptionProcess(result["transcription"], tracer)

            if isinstance(eng, str) and eng.startswith("ERROR"):
                tracer.end()
                ArchSmith.persist()
                return {
                    "error": eng,
                    "image": os.path.basename(image_path),
                    "traditional_chinese": result["transcription"],
                    "simplified_chinese": simCN
                }

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
                "ner_labels": [list(pair) for pair in labels]
            }

        except Exception as e:
            tracer.addReport(ASReport("MMPROCESSOR PIPELINE ERROR", str(e)))
            tracer.end()
            ArchSmith.persist()
            return {"error": f"An error occured in mmProcess: {e}"}