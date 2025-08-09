import os
from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger, Universal, FileOps
from sessionManagement import checkSession
from models import Artefact, User, Batch, BatchProcessingJob, BatchArtefact
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
            Logger.log("DATAIMPORT UPLOAD ERROR: Failed to upload file to FileManager. File path: '{}', original filename: '{}'; error: {}".format(fileObj.identifierPath(), file.filename, res))

            remove = FileManager.removeLocally(fileObj)
            if remove != True:
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to remove file locally after FM save failure. File path: '{}', original filename: '{}'; error: {}".format(fileObj.identifierPath(), file.filename, remove))

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
        batch = Batch(user.id, batchArtefacts=successfulFiles)
        batch.save()
    except Exception as e:
        Logger.log("DATAIMPORT UPLOAD ERROR: Failed to create batch for user '{}' with '{}' artefacts; error: {}".format(user.id, len(successfulFiles), e))

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
    
        if batch.stage != Batch.Stage.UPLOAD_PENDING:
            return JSONRes.new(400, "Batch stage is not upload pending.")
        
        if not isinstance(batch, Batch):
            raise Exception("Unexpected response in loading batch; error: {}".format(batch))

    except Exception as e:
        Logger.log("DATAIMPORT CONFIRMBATCH ERROR: Failed to load batch '{}'; error: {}".format(batchID, e))
        return JSONRes.ambiguousError()

    # Ensure user owns the batch
    if batch.userID != user.id:
        return JSONRes.new(403, "You don't have permission to start this batch's processing.", ResType.USERERROR)

    batch.stage = Batch.Stage.UNPROCESSED
    batch.save()
    
    try:
        batch.initJob()
        batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
        batch.job.save()

        return JSONRes.new(200, "Batch processing started.", batchID=batch.id)

    except Exception as e:
        Logger.log("DATAIMPORT CONFIRMBATCH ERROR: Failed to confirm batch '{}'; error: {}".format(batchID, e))
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
        Logger.log("DATAIMPORT GETALLBATCHES ERROR: Failed to load batches; error: {}".format(e))
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
            Logger.log("DATAIMPORT GETALLBATCHES ERROR: Failed to filter batch '{}'; error: {}".format(b.id, e))
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
        Logger.log("DATAIMPORT GETUSERBATCHES ERROR: Failed to retrieve batches for user '{}'; error: {}".format(user.id, e))
        return JSONRes.ambiguousError()

@dataImportBP.route('/batches/cancel', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str))
@checkSession(strict=True, provideUser=True)
def cancelBatch(user: User):

    batchID: str = request.json.get('batchID')

    try:
        batchJob = BatchProcessingJob.load(batchID)

    except Exception as e:
        Logger.log("DATAIMPORT CANCELBATCH ERROR: Failed to load batch '{}'; error: {}".format(batchID, e))
        return JSONRes.ambiguousError()

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
        Logger.log("DATAIMPORT CANCELBATCH ERROR: Failed to cancel batch '{}'; error: {}".format(batchJob.batchID, e))
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
            return JSONRes.new(404, "Batch '{}' not found.".format(batchID))

    except Exception as e:
        Logger.log("DATAIMPORT DELETE ERROR: Failed to load batch '{}'; error : {}".format(batchID, e))
        return JSONRes.ambiguousError()

    deletionResults = {}
    batchArtefacts = batch.artefacts or {}

    for artefactID in batchArtefacts:

        artefact = batchArtefacts[artefactID].artefact

        # Delete file from FileManager
        res = artefact.fmDelete()
        if res is not True:
            Logger.log("DATAIMPORT DELETE ERROR: Failed to FM delete file for artefact '{}'; error: {}".format(artefactID, res))
            deletionResults[artefactID] = "ERROR: Failed to FM delete artefact."
            continue

        try:
            # Delete DB record
            artefact.destroy()
            deletionResults[artefactID] = "SUCCESS: Artefact deleted successfully."

        except Exception as e:
            Logger.log("DATAIMPORT DELETE ERROR: Failed to DB delete artefact '{}'; error: {}".format(artefactID, e))
            deletionResults[artefactID]= "ERROR: Failed to DB delete artefact."

    # Attempt to delete the batch
    try:
        batch.destroy()
        deletionResults[batchID] = "SUCCESS: Batch deleted successfully."
    except Exception as e:
        Logger.log("DATAIMPORT DELETE ERROR: Failed to delete batch '{}'; error: {}".format(batchID, e))
        deletionResults[batchID] = "ERROR: Failed to delete batch."

    # Determine overall response
    hasErrors = any("ERROR" in msg for msg in deletionResults.values())
    if hasErrors:
        return JSONRes.new(207, "Batch '{}' deletion partially failed.".format(batchID), updates=deletionResults, batchID=batchID)

    return JSONRes.new(200, "Batch '{}' and all artefacts deleted.".format(batchID), updates=deletionResults, batchID=batchID)

@dataImportBP.route('/vetting/start', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str))
@checkSession(strict=True)
def startVetting():
    '''
    Start a vet
    '''
    batchID = request.json.get('batchID')
    try:
        batch = Batch.load(batchID)

        if batch is None:
            return JSONRes.new(404, "Batch not found.")

        if batch.stage != Batch.Stage.PROCESSED:
            return JSONRes.new(400, "Batch must be in 'processed' stage to begin vetting.")

        batch.stage = Batch.Stage.VETTING
        batch.save()

        return JSONRes.new(200, "Batch moved to vetting stage.", batchID=batchID)
    except Exception as e:
        Logger.log("DATAIMPORT VETTING START ERROR: Batch '{}' failed to start vetting; error {}".format(batchID, e))
        return JSONRes.ambiguousError()
    
@dataImportBP.route('/vetting/confirm', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str, 'artefactID', str))
@checkSession(strict=True)
def confirmArtefact():
    '''
    Vet an artefact 
    '''
    batchID = request.json.get('batchID')
    artefactID = request.json.get('artefactID')

    try:
        batch = Batch.load(batchID)

        if batch is None:
            return JSONRes.new(404, "Batch not found.")

        if batch.stage != Batch.Stage.VETTING:
            return JSONRes.new(400, "Batch not in vetting stage.")

        batchArtefact = batch.get(artefactID)
        if not batchArtefact:
            Logger.log("DATAIMPORT VETTING CONFIRM ERROR: Artefact '{}' not found in batch {}".format(artefactID, batchID))
            return JSONRes.new(404, "Artefact not found in batch.")

        batchArtefact.stage = BatchArtefact.Status.CONFIRMED
        batchArtefact.save()

        return JSONRes.new(200, "Artefact confirmed.")
    except Exception as e:
        Logger.log("DATAIMPORT CONFIRMARTEFACT ERROR: Batch '{}' failed to confirm artefact; error {}".format(batchID, e))
        return JSONRes.ambiguousError()

@dataImportBP.route('/integration', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str))
@checkSession(strict=True)
def startIntegration():
    batchID = request.json.get('batchID')
    try:
        batch = Batch.load(batchID)

        if batch is None:
            return JSONRes.new(404, "Batch not found.")
       
        if batch.stage != Batch.Stage.VETTING:
            return JSONRes.new(400, "Batch must be in 'vetting' stage to start integration.")

        allConfirmed = all(a.stage == BatchArtefact.Status.CONFIRMED for a in batch.artefacts.values())
        if not allConfirmed:
            return JSONRes.new(400, "Not all artefacts are confirmed.")

        batch.stage = Batch.Stage.INTEGRATION
        batch.save()

        return JSONRes.new(200, "Batch moved to integration stage.", batchID=batchID)

    except Exception as e:
        Logger.log("DATAIMPORT INTEGRATION ERROR: Batch '{}' failed to integrate; error {}".format(batchID, e))
        return JSONRes.ambiguousError()
    
@dataImportBP.route('/completebatch', methods=['POST'])
@jsonOnly
@enforceSchema(('batchID', str))
@checkSession(strict=True)
def completeBatch():
    batchID = request.json.get('batchID')
    try:
        batch = Batch.load(batchID)

        if batch.stage != Batch.Stage.INTEGRATION:
            return JSONRes.new(400, "Batch must be in 'integration' stage to mark as complete.")

        batch.stage = Batch.Stage.COMPLETED
        batch.save()

        return JSONRes.new(200, "Batch marked as complete.", batchID=batchID)

    except Exception as e:
        Logger.log("DATAIMPORT COMPLETEBATCH ERROR: Batch '{}' failed to complete; error {}".format(batchID, e))
        return JSONRes.ambiguousError()