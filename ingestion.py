import time, functools, copy
from models import Batch, BatchProcessingJob, BatchArtefact, Artefact, Metadata
from services import ThreadManager, Universal
from fm import FileManager, File
from metagen import MetadataGenerator
from services import Logger  

class DataImportProcessor:
    @staticmethod
    def processBatch(batch: Batch, chunkSize: int=10, offloadArtefactPostProcessing: bool=True, maxLife: int=None):
        if maxLife is None:
            maxLife = 60 * 10 # Default to 10 minutes
        start = time.time()
        
        # Start the job
        job = batch.job
        job.start()
        job.save()
        
        targetBatchArtefacts = [x for x in batch.artefacts.values() if x.stage == Batch.Stage.UNPROCESSED]
        
        if len(targetBatchArtefacts) == 0:
            Logger.log("DATAIMPORT PROCESSBATCH: Finished as no unprocessed artefacts found in batch '{}'.".format(batch.id))
            job.end()
            job.save()
            return
        
        Logger.log("DATAIMPORT PROCESSBATCH: Started processing batch '{}' with {} target artefacts and chunk size {}.".format(batch.id, len(targetBatchArtefacts), chunkSize))
        
        chunk_i = 0
        scheduled = 0
        while chunk_i < chunkSize and scheduled < len(targetBatchArtefacts):
            batchArtSequence = targetBatchArtefacts[scheduled : scheduled + chunkSize]
            
            if len(batchArtSequence) > 0:
                ThreadManager.defaultProcessor.addJob(
                    DataImportProcessor.constructHandoffChain(batch.id, batchArtSequence, chunk_i, offloadArtefactPostProcessing)
                )
                Logger.log("DATAIMPORT PROCESSBATCH: Chunk slot {} triggered with {} artefacts.".format(chunk_i, len(batchArtSequence)))
            
            chunk_i += 1
            scheduled += len(batchArtSequence)
        
        while not all(x.stage == Batch.Stage.PROCESSED for x in targetBatchArtefacts):
            if (time.time() - start) > maxLife:
                Logger.log("DATAIMPORT PROCESSBATCH: Max life exceeded for batch '{}'. Stopping processing with {} processed out of {}.".format(batch.id, sum(x.stage == Batch.Stage.PROCESSED for x in targetBatchArtefacts), len(targetBatchArtefacts)))
                job.end()
                job.save()
                return
            
            job.reload() # Refresh the job status. By memory reference, chunks will also terminate if the job is cancelled.
            if job.status == BatchProcessingJob.Status.CANCELLED:
                Logger.log("DATAIMPORT PROCESSBATCH: Job cancelled for batch '{}'. Stopping processing with {} processed out of {}.".format(batch.id, sum(x.stage == Batch.Stage.PROCESSED for x in targetBatchArtefacts), len(targetBatchArtefacts)))
                job.end()
                job.save()
                return
            
            time.sleep(1)
        
        Logger.log("DATAIMPORT PROCESSBATCH: Finished processing batch '{}' with {} target artefacts.".format(batch.id, len(targetBatchArtefacts)))
        job.end()
        job.save()
        return
    
    @staticmethod
    def constructHandoffChain(batchID: str, batchArtSequence: list[BatchArtefact], chainIndex: int, offloadArtefactPostProcessing: bool=True):
        job = BatchProcessingJob.load(batchID)
        
        def scheduleNextProcess():
            if isinstance(job, BatchProcessingJob):
                job.reload() # refresh job status
                if job.status == BatchProcessingJob.Status.CANCELLED:
                    Logger.log("DATAIMPORT SCHEDULENEXTPROCESS: {} {}: Job cancelled for batch '{}'. Terminating chain with {} remaining artefacts.".format(batchID, chainIndex, batchID, len(batchArtSequence)))
                    return
            
            if len(batchArtSequence) > 0:
                nextBatchArt = batchArtSequence.pop(0)
                ThreadManager.defaultProcessor.addJob(DataImportProcessor.processItem, nextBatchArt, offloadArtefactPostProcessing, scheduleNextProcess)

        return scheduleNextProcess
    
    @staticmethod
    def endProcessing(batchArt: BatchArtefact, fmFile: File, processingStartTime: float, errorMsg: str=None, offloadArtefactPostProcessing: bool=True, scheduleNextProcess=None):
        if offloadArtefactPostProcessing and isinstance(fmFile, File):
            res = FileManager.offload(fmFile)
            if isinstance(res, str):
                Logger.log("DATAIMPORT ENDPROCESSING WARNING: Non-success response in offload attempt for artefact '{}' of batch '{}' (file: '{}'); response: {}".format(batchArt.artefactID, batchArt.batchID, fmFile.identifierPath(), res))
        
        batchArt.processedTime = Universal.utcNowString()
        batchArt.processedDuration = '{:.3f}'.format(time.time() - processingStartTime)
        batchArt.processingError = errorMsg
        batchArt.stage = Batch.Stage.PROCESSED
        batchArt.artefact = None # Clear the artefact reference to free memory
        batchArt.save()
        
        if callable(scheduleNextProcess):
            scheduleNextProcess()
    
    @staticmethod
    def processItem(batchArtefact: BatchArtefact, offloadArtefactPostProcessing: bool=True, scheduleNextProcess=None):
        if batchArtefact.stage != Batch.Stage.UNPROCESSED:
            Logger.log("DATAIMPORT PROCESSITEM WARNING: Processing Artefact '{}' of batch '{}' despite stage of '{}'.".format(batchArtefact.artefactID, batchArtefact.batchID, batchArtefact.stage.value))
        
        processingStartTime = time.time()
        
        if not isinstance(batchArtefact.artefact, Artefact):
            try:
                batchArtefact.getArtefact(includeMetadata=True)
            except Exception as e:
                Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to obtain Artefact object (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
                DataImportProcessor.endProcessing(batchArtefact, None, processingStartTime, str(e), offloadArtefactPostProcessing, scheduleNextProcess)
                return
        
        fmFile = FileManager.prepFile(file=batchArtefact.artefact.getFMFile())
        if not isinstance(fmFile, File):
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to prepare file for Artefact (BatchID: {}, ArtefactID: {}); FM response: {}".format(batchArtefact.batchID, batchArtefact.artefactID, fmFile))
            DataImportProcessor.endProcessing(batchArtefact, None, processingStartTime, fmFile, offloadArtefactPostProcessing, scheduleNextProcess)
            return
        
        out = None
        try:
            out = MetadataGenerator.generate(fmFile.path())
            if isinstance(out, str):
                raise Exception(out)
        except Exception as e:
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failure in metadata generation (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
            DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, str(e), offloadArtefactPostProcessing, scheduleNextProcess)
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
            DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, str(e), offloadArtefactPostProcessing, scheduleNextProcess)
        
        batchArtefact.artefact.save()
        
        Logger.log("DATAIMPORT PROCESSITEM: Artefact '{}' of batch '{}' processed.".format(batchArtefact.artefactID, batchArtefact.batchID))
        
        DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, errors, offloadArtefactPostProcessing, scheduleNextProcess)
        return