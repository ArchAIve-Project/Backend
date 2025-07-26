import os
from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger, Universal, FileOps
from sessionManagement import checkSession
from models import Artefact, User, Batch, BatchProcessingJob
from fm import FileManager, File
from services import ThreadManager
from ingestion import DataImportProcessor
from services import Logger

dataimportBP = Blueprint('dataimport', __name__, url_prefix="/dataimport")

@dataimportBP.route('/upload', methods=['POST'])
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
    if not files or all(file.filename == '' for file in files):
        return JSONRes.new(400, "No files selected.")

    fileSaveUpdates = {}
    sucessfulFiles = []
    processedNames = set()

    for file in files:
        if file and file.filename: 
            
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

            # Check file size
            validSize = FileOps.fileSize(file)
            if not validSize:
                fileSaveUpdates[file.filename] = "ERROR: File size > 10 MB."
                continue
           
            # Save file locally
            filename = Universal.generateUniqueID()
            filename = "{}.{}".format(filename, FileOps.getFileExtension(file.filename))
            fileObj = File(filename, 'artefacts')
    
            try:
                file.save(fileObj.path())
            except Exception as e:
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save file: {}".format(e))
                fileSaveUpdates[file.filename] = "ERROR: Failed to save file."
                continue

            # Sync local to firebase 
            res = FileManager.save(file=fileObj)
            if isinstance(res, str): # if fail
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save file: {}. ".format(res))
                
                remove = FileManager.removeLocally(fileObj) 
                if remove != True:
                    Logger.log("DATAIMPORT UPLOAD ERROR: Failed to remove file: {}".format(remove))
                
                fileSaveUpdates[file.filename] = "ERROR: Failed to save file."
                continue

            # Create artefact
            artefact = Artefact.fromFMFile(fileObj)
            
            try:
                artefact.save()
                sucessfulFiles.append(artefact)
                fileSaveUpdates[file.filename] = "File uploaded successfully."

            except Exception as e:
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save artefact for {}: {}".format(file.filename, str(e)))

                FileManager.delete(file=fileObj)

                fileSaveUpdates[file.filename] = "ERROR: Failed to save artefact."
                continue
    
    if not sucessfulFiles:
        return JSONRes.new(400, "No artefacts were successfully uploaded.", ResType.ERROR, updates=fileSaveUpdates)
    
    try:
        # Create batch
        batch = Batch(user.id, batchArtefacts=sucessfulFiles)
        batch.save()

    except Exception as e:
        Logger.log("DATAIMPORT UPLOAD ERROR: Failed to create batch: {}".format(e))

    return JSONRes.new(200, "Upload results.", ResType.SUCCESS, updates=fileSaveUpdates, batchID=batch.id)

@dataimportBP.route('/confirm', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def confirm(user: User):
    """
    Confirms a previously created upload batch and triggers processing.

    Expected JSON:
    {"batchID": "<batch_id>"}

    Steps:
    - Validate batchID and user ownership.
    - Initialize and start the batch processing job.

    Return batchID and jobID.
    """
    data = request.get_json(silent=True)
    if not data or "batchID" not in data:
        return JSONRes.new(400, "batchID is required.")
    
    batchID = data["batchID"]

    # Load batch
    batch = Batch.load(batchID)
    if not batch:
        return JSONRes.new(404, "Batch not found.")
    
    # Ensure user owns the batch
    if batch.userID != user.id:
        return JSONRes.new(403, "You do not have permission for this batch.", ResType.ERROR)
    
    try:
        batch.initJob()
        batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
        batch.job.save()
        batch.save()

        return JSONRes.new(200, "message", ResType.SUCCESS, batchID=batch.id, jobID=batch.job.jobID)
    
    except Exception as e:
        Logger.log("DATAIMPORT CONFIRM ERROR: Failed to confirm batch {}: {}".format(batchID, e))
        return JSONRes.new(500, "Failed to start batch processing.", ResType.ERROR)

@dataimportBP.route('/batches', methods=['GET'])
@checkSession(strict=True)
def getAllBatches():
    try:
        batches = Batch.load()
        if not batches:
            return JSONRes.new(200, "No batches found.")

        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "All batches retrieved.", batches=formatted)

    except Exception as e:
        return JSONRes.ambiguousError(str(e))

@dataimportBP.route('/userbatches', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def getUserBatches(user: User):
    try:
        batches = Batch.load(userID=user.id, withArtefacts=False)
        if not batches:
            return JSONRes.new(200, "No batches found.")

        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "User batches retrieved.", batches=formatted)

    except Exception as e:
        return JSONRes.ambiguousError(str(e))

@dataimportBP.route('/batches/<batchID>/cancel', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def cancelBatch(user: User, batchID: str):
    try:
        batch = Batch.load(id=batchID)
        if not batch:
            return JSONRes.new(404, "Batch not found.")

        # Ensure batch belongs to user or has permission
        if batch.userID != user.id:
            return JSONRes.unauthorised()

        job = batch.job
        if job.status in [BatchProcessingJob.Status.COMPLETED]:
            return JSONRes.new(400, "Batch job already completed")
        elif job.status in [BatchProcessingJob.Status.CANCELLED]:
            return JSONRes.new(200, "Batch job already cancelled.")

        job.cancel()
        job.save()

        return JSONRes.new(200, "Batch {} processing cancelled.".format(batchID))

    except Exception as e:
        return JSONRes.ambiguousError(str(e))

@dataimportBP.route('/batches/<batchID>/delete', methods=['DELETE'])
@checkSession(strict=True, provideUser=True)
def deleteBatch(user: User, batchID: str):
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

        if batch.userID != user.id:
            return JSONRes.unauthorised()

        deletionResults = {}
        artefacts = getattr(batch, "artefacts", [])

        for artefactRef in artefacts:
            artefactID = artefactRef if isinstance(artefactRef, str) else getattr(artefactRef, 'id', None)
            if not artefactID:
                deletionResults["unknown_artefact"] = "Skipped invalid artefact reference with no ID."
                continue

            try:
                artefact = artefactRef if isinstance(artefactRef, Artefact) else Artefact.load(id=artefactID)
                if not artefact:
                    deletionResults[artefactID] = "ERROR: Artefact not found."
                    continue

                # Delete file from FileManager
                res = artefact.fmDelete()
                if res is not True:
                    deletionResults[artefactID] = "ERROR: Failed to delete file."
                    continue

                # Delete DB record
                try:
                    artefact.destroy()
                    deletionResults[artefactID] = "Artefact deleted successfully."

                except Exception as e:
                    Logger.log("DATAIMPORT DELETE ERROR: Failed to delete artefact DB record {}: {}".format(artefactID, e))
                    deletionResults[artefactID] = "ERROR: Failed to delete artefact DB record."

            except Exception as e:
                Logger.log("DATAIMPORT DELETE ERROR: Failed to delete artefact {}: {}".format(artefactID, e))
                deletionResults[artefactID] = "ERROR: Unexpected failure during artefact deletion."

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
        return JSONRes.ambiguousError(str(e))