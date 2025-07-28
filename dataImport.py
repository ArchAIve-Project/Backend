import os
from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger, Universal, FileOps
from sessionManagement import checkSession
from models import Artefact, User, Batch, BatchProcessingJob
from fm import FileManager, File
from services import ThreadManager
from ingestion import DataImportProcessor
from services import Logger, Universal
from decorators import jsonOnly, enforceSchema

dataImportBP = Blueprint('dataImport', __name__, url_prefix="/dataImport")

@dataImportBP.route('/upload', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def upload(user : User):
    """
    Handles artefact file uploads.

    For each file:
    - Validates type and size
    - Saves locally
    - Uploads to Firebase
    - Creates DB record
    
    Creates a Batch containing all successfully uploaded Artefacts.
    If no files were successfully uploaded, no batch is created.

    On failure at any step, the file is skipped and the error is logged and returned.

    Returns a summary of upload results per file and batchid.
    """
    if 'file' not in request.files:
        return JSONRes.new(400, "No file part in request.")

    files = request.files.getlist('file')
    files = list(filter(lambda f: f and f.filename, files)) 
    if not files:
        return JSONRes.new(400, "No files selected.", ResType.USERERROR)

    fileSaveUpdates = {}
    successfulFiles = []
    processedNames = set()

    for file in files:
        
        fileName = os.path.splitext(file.filename)[0].lower()

        # Check duplicate in current request
        if fileName in processedNames:
            fileSaveUpdates[file.filename] = "ERROR: Duplicate file."
            continue

        processedNames.add(fileName)

        # Check file type
        if not FileOps.allowedFileExtension(file.filename):
            fileSaveUpdates[file.filename] = "ERROR: File type not allowed."
            continue

        # Get file size and check file size
        validSize = FileOps.getFileStorageSize(file)
        if validSize is None:
            fileSaveUpdates[file.filename] = "ERROR: Failed to get file size."
            continue

        if validSize > Universal.MAX_FILE_SIZE:
            fileSaveUpdates[file.filename] = "ERROR: File size exceeds 10MB."
            continue
        
        # Save file locally
        filename = Universal.generateUniqueID()
        filename = "{}.{}".format(filename, FileOps.getFileExtension(file.filename))
        fileObj = File(filename, 'artefacts')

        try:
            file.save(fileObj.path())
        except Exception as e:
            Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save file '{}' with original name '{}'; error: {}".format(fileObj.identifierPath(), file.filename, e))
            fileSaveUpdates[file.filename] = "ERROR: Failed to save file."
            continue

        # Sync local to firebase 
        res = FileManager.save(file=fileObj)
        if isinstance(res, str):
            Logger.log("DATAIMPORT UPLOAD ERROR: Failed to upload file to FileManager. File path: '{}', original filename: '{}'. Error: {}".format(fileObj.identifierPath(), file.filename, res))
                
            remove = FileManager.removeLocally(fileObj)
            if remove != True: # failed to remove locally after FM save failure
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to remove locally file after FM save failure. File path: '{}', original filename: '{}'. Remove result: {}".format(fileObj.identifierPath(), file.filename, remove))
            
            fileSaveUpdates[file.filename] = "ERROR: Failed to save file."
            continue

        # Create artefact
        artefact = Artefact.fromFMFile(fileObj)
        
        artefact.save()
        successfulFiles.append(artefact)
        fileSaveUpdates[file.filename] = "SUCCESS: File uploaded successfully."
    
    if not successfulFiles:
        return JSONRes.new(400, "No artefacts were processed successfully.", updates=fileSaveUpdates)
    
    try:
        # Create batch
        batch = Batch(user.id, batchArtefacts=successfulFiles)
        batch.save()

    except Exception as e:
        Logger.log("DATAIMPORT UPLOAD ERROR: Failed to create batch for user '{}' with {} artefacts. Error: {}".format(user.id, len(successfulFiles), e))

    return JSONRes.new(200, "Upload results.", updates=fileSaveUpdates, batchID=batch.id)

@dataImportBP.route('/confirm', methods=['POST'])
@jsonOnly 
@enforceSchema( ('batchID', str) )
@checkSession(strict=True, provideUser=True)
def confirmBatch(user: User):
    """
    Confirms a previously created upload batch and triggers processing.

    Expected JSON:
    {"batchID": "<batch_id>"}

    Steps:
    - Validate batchID and user ownership.
    - Initialize and start the batch processing job.

    Return batchID and jobID.
    """
    batchID: str = request.json.get('batchID')

    # Load batch
    batch = None
    try:
        batch = Batch.load(batchID)
        if batch is None:
            return JSONRes.new(404, "Batch not found.")
        if not isinstance(batch, Batch):
            raise Exception("Unexpected response in loading batch; response: {}".format(batch))
    
    except Exception as e:
        Logger.log("DATAIMPORT CONFIRMBATCH ERROR: Failed to load batch {}; error: {}".format(batchID, e))
        return JSONRes.ambiguousError()

    # Ensure user owns the batch
    if batch.userID != user.id:
        return JSONRes.new(403, "You don't have permission to start this batch's processing.", ResType.USERERROR)
    
    try:
        batch.initJob()
        batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
        batch.job.save()

        return JSONRes.new(200, "Batch processing started.", batchID=batch.id)
    
    except Exception as e:
        Logger.log("DATAIMPORT CONFIRMBATCH ERROR: Failed to confirm batch {}: {}".format(batchID, e))
        return JSONRes.ambiguousError()

@dataImportBP.route('/batches', methods=['GET'])
@checkSession(strict=True)
def getAllBatches():
    try:
        batches = Batch.load()
        if not batches:
            return JSONRes.new(200, "No batches found.", batches={})
        if not isinstance(batches, list):
            raise Exception("Expected list of batches, got {}".format(type(batches)))
    
    except Exception as e:
        Logger.log("DATAIMPORT GETALLBATCHES ERROR: {}".format(e))
        return JSONRes.ambiguousError()
    
    # filter out sensitive information
    formatted = {}
    for b in batches:
        try:
            rep = b.represent()

            # Remove userID completely
            rep.pop("userID", None)

            # Remove processingError from each artefact
            for artefactData in rep.get("artefacts", {}).values():
                artefactData.pop("processingError", None)

            # Keep job but remove jobID
            if "job" in rep and isinstance(rep["job"], dict):
                rep["job"].pop("jobID", None)

            formatted[b.id] = rep

        except Exception as e:
            Logger.log("DATAIMPORT GETALLBATCHES ERROR: Failed to load batch {}: {}".format(b.id, e))
            continue  

    return JSONRes.new(200, "All batches retrieved.", batches=formatted)

@dataImportBP.route('/userBatches', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def getUserBatches(user: User):
    try:
        batches = Batch.load(userID=user.id)
        
        formatted = {b.id: b.represent() for b in batches} if batches else {}
        return JSONRes.new(200, "User batches retrieved.", batches=formatted)

    except Exception as e:
        Logger.log("DATAIMPORT GETUSERBATCHES ERROR: {}".format(e))
        return JSONRes.ambiguousError()

@dataImportBP.route('/batches/cancel', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str)) 
@checkSession(strict=True, provideUser=True)
def cancelBatch(user: User):

    batchID: str = request.json.get('batchID')

    # Now load the job
    batchJob = BatchProcessingJob.load(batchID)
    if not batchJob:
        return JSONRes.new(404, "Batch job not found.")

    if batchJob.status == BatchProcessingJob.Status.COMPLETED:
        return JSONRes.new(400, "Batch job already completed")
    elif batchJob.status == BatchProcessingJob.Status.CANCELLED:
        return JSONRes.new(200, "Batch job already cancelled.")

    try:
        batchJob.cancel()
        return JSONRes.new(200, "Batch processing cancelled.", batchID=batchID)

    except Exception as e:
        Logger.log("DATAIMPORT CANCELBATCH ERROR: {}".format(e))
        return JSONRes.ambiguousError()

@dataImportBP.route('/batches/<batchID>', methods=['DELETE'])
@checkSession(strict=True)
def deleteBatch(batchID: str):
    """
    Deletes a batch and its artefacts.

    For each artefact:
    - Attempts to delete from FileManager
    - Attempts to delete DB record
    - Logs and records errors if any step fails

    Returns a summary of deletion results per artefact and batchID.
    """
    try:
        batch = Batch.load(id=batchID, withArtefacts=True)
        if not batch:
            return JSONRes.new(404, "Batch {} not found.".format(batchID))

        deletionResults = {}
        batchArt = batch.artefacts or {}

        for artefactID in batchArt:

            artefact = batchArt[artefactID].artefact
            if not artefact:
                deletionResults[artefactID] = "ERROR: Artefact not found."
                continue

            # Delete file from FileManager
            res = artefact.fmDelete()
            if res is not True: # Add log
                deletionResults[artefactID] = "ERROR: Failed to delete file."
                continue

            # Delete DB record
            artefact.destroy()
            deletionResults[artefactID] = "SUCCESS: Artefact deleted successfully."
        
        # Attempt to delete the batch
        try:
            batch.destroy()
            deletionResults["batch"] = "Batch deleted successfully."
        except Exception as e:
            Logger.log("DATAIMPORT DELETE ERROR: Failed to delete batch {}: {}".format(batchID, e))
            deletionResults["batch"] = "ERROR: Failed to delete batch."

        # Determine overall response
        hasErrors = any("ERROR" in msg for msg in deletionResults.values())
        if hasErrors:
            return JSONRes.new(207, "Batch {} deletion partially failed.".format(batchID), updates=deletionResults, batchID=batchID)

        return JSONRes.new(200, "Batch {} and all artefacts deleted.".format(batchID), updates=deletionResults, batchID=batchID)

    except Exception as e:
        Logger.log("DATAIMPORT DELETE ERROR: {}".format(e))
        return JSONRes.ambiguousError()