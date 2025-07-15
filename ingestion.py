import time
from models import Batch, BatchProcessingJob, BatchArtefact, Artefact, Metadata
from services import ThreadManager, Universal
from metagen import MetadataGenerator
from services import Logger

class DataImportProcessor:
    @staticmethod
    def processBatch(batch: Batch):
        # Start the job
        job = batch.job
        job.start()
        job.save()
        
        # Pick out all artefacts to be processed
        targetArtefacts = [x for x in batch.artefacts.values() if x.stage == Batch.Stage.UNPROCESSED]
        
        
        
        pass
    
    @staticmethod
    def endProcessing(batchArt: BatchArtefact, processingStartTime: float, errorMsg: str=None, updateStageIfError: bool=False):
        batchArt.processedTime = Universal.utcNowString()
        batchArt.processedDuration = '{:.3f}'.format(time.time() - processingStartTime)
        batchArt.processingError = errorMsg
        
        if (errorMsg is None) or (errorMsg != None and updateStageIfError):
            batchArt.stage = Batch.Stage.PROCESSED
        
        batchArt.save()
    
    @staticmethod
    def processItem(batchArtefact: BatchArtefact):
        if batchArtefact.stage != Batch.Stage.UNPROCESSED:
            Logger.log("DATAIMPORT PROCESSITEM WARNING: Processing Artefact '{}' of batch '{}' despite stage of '{}'.".format(batchArtefact.artefactID, batchArtefact.batchID, batchArtefact.stage.value))
        
        processingStartTime = time.time()
        
        if not isinstance(batchArtefact.artefact, Artefact):
            try:
                batchArtefact.getArtefact(includeMetadata=True)
            except Exception as e:
                Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to obtain Artefact object (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
                DataImportProcessor.endProcessing(batchArtefact, processingStartTime, e)
                return
        
        out = None
        try:
            out = MetadataGenerator.generate(batchArtefact.artefact.getFMFile().path())
            if isinstance(out, str):
                raise Exception(out)
        except Exception as e:
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failure in metadata generation (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
            DataImportProcessor.endProcessing(batchArtefact, processingStartTime, str(e))
            return

        errors = None
        if 'errors' in out:
            if isinstance(out['errors'], str):
                errors = out['errors']
            elif isinstance(out['errors'], list):
                errors = ", ".join([e for e in out['errors'] if isinstance(e, str)])
        
        try:
            batchArtefact.artefact.metadata = Metadata.fromMetagen(batchArtefact.artefactID, out)
        except Exception as e:
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to construct Metadata object from generator output (BatchID: {}, ArtefactID: {}); error: {}".format(e))
            DataImportProcessor.endProcessing(batchArtefact, processingStartTime, str(e))
        
        batchArtefact.artefact.save()
        DataImportProcessor.endProcessing(batchArtefact, processingStartTime, errors, updateStageIfError=True)
        
        Logger.log("DATAIMPORT PROCESSITEM: Artefact '{}' of batch '{}' processed.".format(batchArtefact.artefactID, batchArtefact.batchID))
        return
    
    
    # @staticmethod
    # def processBatch(batch: Batch, chunk_size=10):
    #     print("Processing batch: {}".format(batch.id))
    #     batch.getArtefacts(unprocessed=True)
    #     unprocessedItems = [a for a in batch.unprocessed.values() if isinstance(a, Artefact)]
    #     total = len(unprocessedItems)
    #     i = 0
    #     failed_ids = set()

    #     while i < total:
    #         currentChunk = unprocessedItems[i:i + chunk_size]
    #         ThreadManager.defaultProcessor.addJob(ImportProcessor.processChunk, currentChunk, batch, failed_ids)

    #         # Wait until all items in currentChunk are either processed or failed
    #         while not all(
    #             artefact.id in batch.processed or artefact.id in failed_ids
    #             for artefact in currentChunk if isinstance(artefact, Artefact)
    #         ) and not batch.cancelled:
    #             batch.reload()
    #             time.sleep(0.5)

    #         if batch.cancelled:
    #             print("Batch processing cancellation detected.")
    #             break
    #         i += chunk_size
        
    #     batch.job = None
    #     print("Batch processing completed. Processed {} items, failed {} items.".format(
    #         len(batch.processed), len(failed_ids)
    #     ))
    #     return True

    # @staticmethod
    # def processChunk(chunk: list[Artefact], batch: Batch, failed_ids: set):
    #     for artefact in chunk:
    #         try:
    #             ImportProcessor.processItem(artefact)
    #             batch.reload()
    #             batch.move(artefact, Batch.Stage.PROCESSED)
    #             batch.save()
    #         except Exception as e:
    #             Logger.log(f"Failed to process artefact {artefact.id}: {e}")
    #             failed_ids.add(artefact.id)

    #         if batch.cancelled:
    #             print("Batch processing cancelled during chunk processing.")
    #             return

    # @staticmethod
    # def processItem(artefact: Artefact):
    #     out = MetadataGenerator.generate(artefact.getFMFile().path())
    #     if isinstance(out, str):
    #         raise Exception("IMPORTPROCESSOR PROCESSITEM ERROR: Unexpected output from MetadataGenerator when processing artefact '{}'; response: {}".format(artefact.id, out))
    #     artefact.metadata = Metadata.fromMetagen(artefact.id, out)
    #     artefact.save()
    #     return artefact