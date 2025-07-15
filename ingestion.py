import time
from models import Batch, BatchProcessingJob, BatchArtefact, Artefact, Metadata
from services import ThreadManager, Universal
from metagen import MetadataGenerator
from services import Logger

class DataImportProcessor:
    @staticmethod
    def processBatch(batch: Batch, chunkSize: int=10):
        # Start the job
        job = batch.job
        job.start()
        job.save()
        
        # Pick out all artefacts to be processed
        targetArtefacts = [x for x in batch.artefacts.values() if x.stage == Batch.Stage.UNPROCESSED]
        targetCount = len(targetArtefacts)
        Logger.log("DATAIMPORT PROCESSBATCH: Processing batch '{}' with '{}' targets and chunk size '{}'.".format(batch.id, targetCount, chunkSize))
        
        # Chunk
        completionCount = 0
        
        chunk = []
        while len(targetArtefacts) > 0:
            chunk = targetArtefacts[:chunkSize]
            
            for batchArt in chunk:
                ThreadManager.defaultProcessor.addJob(DataImportProcessor.processItem, batchArt)
            
            while not all(batchArt.stage == Batch.Stage.PROCESSED for batchArt in chunk) and not job.status == BatchProcessingJob.Status.CANCELLED:
                job.reload()
                time.sleep(0.5)
            
            if job.status == BatchProcessingJob.Status.CANCELLED:
                Logger.log("DATAIMPORT PROCESSBATCH: Processing of batch '{}' cancelled. Completion: {} out of {}".format(batch.id, completionCount, targetCount))
                job.end()
                job.save()
                return
            
            completionCount += len(chunk)
            targetArtefacts = targetArtefacts[chunkSize:]

        Logger.log("DATAIMPORT PROCESSBATCH: Processed '{}' artefacts for batch '{}' out of '{}'.".format(completionCount, batch.id, targetCount))
        job.end()
        job.save()
    
    @staticmethod
    def endProcessing(batchArt: BatchArtefact, processingStartTime: float, errorMsg: str=None):
        batchArt.processedTime = Universal.utcNowString()
        batchArt.processedDuration = '{:.3f}'.format(time.time() - processingStartTime)
        batchArt.processingError = errorMsg
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
        DataImportProcessor.endProcessing(batchArtefact, processingStartTime, errors)
        
        Logger.log("DATAIMPORT PROCESSITEM: Artefact '{}' of batch '{}' processed.".format(batchArtefact.artefactID, batchArtefact.batchID))
        return