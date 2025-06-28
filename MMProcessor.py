from ccrPipeline import CCRPipeline
from transcriptionProcessor import TranscriptionProcessor
from addons import ASTracer, ASReport
from NERPipeline import NERPipeline
import os

class MMProcessor:
    '''
    MMProcessor orchestrates the full OCR + NLP pipeline.

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
            tracer.addReport(ASReport("MMPROCESSOR CCRPROCESS ERROR", msg))
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
                if isinstance(simplified, str) and simplified.startswith("ERROR:"):
                    tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTIONPROCESS SIMPLIFIED ERROR", simplified))

                if isinstance(english, str) and english.startswith("ERROR:"):
                    tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTIONPROCESS ENGLISH ERROR", english))

                if isinstance(summary, str) and summary.startswith("ERROR:"):
                    tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTIONPROCESS SUMMARY ERROR", summary))
                    
                return simplified, english, summary

            tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTIONPROCESS ERROR", "Unexpected result structure", result))
            return traditional_text, "ERROR: Translation failed", None

        except Exception as e:
            msg = f"Exception during transcription process: {e}"
            tracer.addReport(ASReport("MMPROCESSOR TRANSCRIPTIONPROCESS ERROR", msg))
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
                tracer.addReport(ASReport("MMPROCESSOR NERPROCESS ERROR", result))
                return []
            return result

        except Exception as e:
            msg = f"Exception during NER prediction: {e}"
            tracer.addReport(ASReport("MMPROCESSOR NERPROCESS ERROR", msg))
            return []

    @staticmethod
    def mmProcess(image_path: str, tracer: ASTracer, gt_path: str = None):
        """
        Executes the complete multi-modal pipeline:
        - Performs OCR and optional correction.
        - Converts and translates the recognized text.
        - Extracts named entities from the English output.

        Returns:
            dict: Final pipeline output with all intermediate results and safe fallbacks.
        """
        errors = []
        tradCNText = simCN = eng = summary = None
        corrected = pre_acc = post_acc = None
        labels = None

        try:
            # OCR
            try:
                ccr_output = MMProcessor.ccrProcess(image_path=image_path, tracer=tracer, gt_path=gt_path)
                tradCNText = ccr_output.get("transcription")
                corrected = ccr_output.get("corrected", False)
                pre_acc = ccr_output.get("preCorrectionAccuracy")
                post_acc = ccr_output.get("postCorrectionAccuracy")

                if isinstance(tradCNText, str) and tradCNText.startswith("ERROR"):
                    errors.append(tradCNText)
                    tradCNText = None
            except Exception as e:
                msg = f"OCR sub-process crashed: {e}"
                tracer.addReport(ASReport("MMPROCESSOR MMPROCESS ERROR", msg))
                errors.append(f"ERROR: {msg}")

            # Transcription
            if tradCNText:
                try:
                    simCN, eng, summary = MMProcessor.transcriptionProcess(tradCNText, tracer)

                    if isinstance(simCN, str) and simCN.startswith("ERROR"):
                        errors.append(simCN)
                        simCN = None
                    if isinstance(eng, str) and eng.startswith("ERROR"):
                        errors.append(eng)
                        eng = None
                    if isinstance(summary, str) and summary.startswith("ERROR"):
                        errors.append(summary)
                        summary = None
                except Exception as e:
                    msg = f"Transcription sub-process crashed: {e}"
                    tracer.addReport(ASReport("MMPROCESSOR MMPROCESS ERROR", msg))
                    errors.append(f"ERROR: {msg}")

            # NER
            if summary:
                try:
                    ner_output = MMProcessor.nerProcess(summary, tracer)
                    if isinstance(ner_output, str) and ner_output.startswith("ERROR"):
                        errors.append(ner_output)
                    else:
                        labels = [list(pair) for pair in ner_output]
                except Exception as e:
                    msg = f"NER sub-process crashed: {e}"
                    tracer.addReport(ASReport("MMPROCESSOR MMPROCESS ERROR", msg))
                    errors.append(f"ERROR: {msg}")

            # Report if anything failed
            if errors:
                tracer.addReport(ASReport(
                    source="MMPROCESSOR PROCESS ERROR",
                    message="One or more sub-processes failed.",
                    extraData={"msgs": errors}
                ))

            return {
                "image": os.path.basename(image_path),
                "traditional_chinese": tradCNText,
                "correction_applied": corrected,
                "pre_correction_accuracy": pre_acc,
                "post_correction_accuracy": post_acc,
                "simplified_chinese": simCN,
                "english_translation": eng,
                "summary": summary,
                "ner_labels": labels,
                "errors": errors if errors else None
            }

        except Exception as e:
            tracer.addReport(ASReport("MMPROCESSOR MMPROCESS ERROR", str(e)))
            return {"error": f"An error occured in mmProcess: {e}"}