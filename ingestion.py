import time, functools
from models import Batch, BatchProcessingJob, BatchArtefact, Artefact, Metadata
from decorators import debug
from services import ThreadManager, Universal
from fm import FileManager, File
from metagen import MetadataGenerator
from services import Logger  

class DataImportProcessor:
    @staticmethod
    @debug
    def processBatch(batch: Batch, chunkSize: int=10, offloadArtefactPostProcessing: bool=True):
        try:
            # Start the job
            job = batch.job
            job.start()
            job.save()
            
            targetBatchArtefacts = [x for x in batch.artefacts.values() if x.stage == Batch.Stage.UNPROCESSED]
            chunk: list[BatchArtefact] = []
            
            targetCount = len(targetBatchArtefacts)
            completionCount = 0
            
            processingComplete = False
            
            Logger.log("DATAIMPORT PROCESSBATCH: Processing batch '{}' with '{}' targets and chunk size '{}'.".format(batch.id, targetCount, chunkSize))
            
            while (len(targetBatchArtefacts) > 0) or (not processingComplete):
                processedChunkIndices = [i for i, x in enumerate(chunk) if x.stage == Batch.Stage.PROCESSED]
                for i in processedChunkIndices:
                    try:
                        chunk.remove(chunk[i])
                    except:
                        pass # Ignore if the index is out of range or already removed
                    completionCount += 1
                
                if len(chunk) == 0 and len(targetBatchArtefacts) == 0:
                    processingComplete = True
                    continue
                
                if len(chunk) < chunkSize and len(targetBatchArtefacts) > 0: # if the chunk is not full, add more artefacts
                    batchArt = targetBatchArtefacts.pop(0)
                    chunk.append(batchArt)
                    ThreadManager.defaultProcessor.addJob(DataImportProcessor.processItem, batchArt, offloadArtefactPostProcessing)
                
                job.reload() # reload the job to check its status
                if job.status == BatchProcessingJob.Status.CANCELLED:
                    Logger.log("DATAIMPORT PROCESSBATCH: Processing of batch '{}' cancelled. Completion: {} out of {}".format(batch.id, completionCount, targetCount))
                    job.end()
                    job.save()
                    return
                time.sleep(0.5) # Sleep to avoid busy-waiting
            
            Logger.log("DATAIMPORT PROCESSBATCH: Processed '{}' artefacts for batch '{}' out of '{}'.".format(completionCount, batch.id, targetCount))
            job.end()
            job.save()
            return
        except Exception as e:
            Logger.log("DATAIMPORT PROCESSBATCH ERROR: Failed to process batch '{}'; error: {}".format(batch.id, e))
            return
    
    @staticmethod
    def endProcessing(batchArt: BatchArtefact, fmFile: File, processingStartTime: float, errorMsg: str=None, offloadArtefactPostProcessing: bool=True):
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
    
    @staticmethod
    def processItem(batchArtefact: BatchArtefact, offloadArtefactPostProcessing: bool=True):
        if batchArtefact.stage != Batch.Stage.UNPROCESSED:
            Logger.log("DATAIMPORT PROCESSITEM WARNING: Processing Artefact '{}' of batch '{}' despite stage of '{}'.".format(batchArtefact.artefactID, batchArtefact.batchID, batchArtefact.stage.value))
        
        processingStartTime = time.time()
        
        if not isinstance(batchArtefact.artefact, Artefact):
            try:
                batchArtefact.getArtefact(includeMetadata=True)
            except Exception as e:
                Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to obtain Artefact object (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
                DataImportProcessor.endProcessing(batchArtefact, None, processingStartTime, str(e))
                return
        
        fmFile = FileManager.prepFile(file=batchArtefact.artefact.getFMFile())
        if not isinstance(fmFile, File):
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failed to prepare file for Artefact (BatchID: {}, ArtefactID: {}); FM response: {}".format(batchArtefact.batchID, batchArtefact.artefactID, fmFile))
            DataImportProcessor.endProcessing(batchArtefact, None, processingStartTime, fmFile)
            return
        
        out = None
        try:
            out = MetadataGenerator.generate(fmFile.path())
            if isinstance(out, str):
                raise Exception(out)
        except Exception as e:
            Logger.log("DATAIMPORT PROCESSITEM ERROR: Failure in metadata generation (BatchID: {}, ArtefactID: {}); error: {}".format(batchArtefact.batchID, batchArtefact.artefactID, e))
            DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, str(e), offloadArtefactPostProcessing)
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
            DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, str(e), offloadArtefactPostProcessing)
        
        batchArtefact.artefact.save()
        DataImportProcessor.endProcessing(batchArtefact, fmFile, processingStartTime, errors, offloadArtefactPostProcessing)
        
        Logger.log("DATAIMPORT PROCESSITEM: Artefact '{}' of batch '{}' processed.".format(batchArtefact.artefactID, batchArtefact.batchID))
        return