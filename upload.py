import os, time
from werkzeug.utils import secure_filename
from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from services import Logger, Encryption, Universal, FileOps
from sessionManagement import checkSession
from models import Metadata, Artefact, User, Batch, BatchArtefact, BatchProcessingJob
from fm import FileManager, File
from services import ThreadManager
from metagen import MetadataGenerator
from ingestion import DataImportProcessor
from services import Logger

dataimportBP = Blueprint('dataimport', __name__, url_prefix="/dataimport")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dataimportBP.route('/upload', methods=['POST'])
@checkSession(strict=True)
def upload():
    """
    Handles artefact file uploads.

    For each file:
    - Validates type and size
    - Saves locally
    - Uploads to Firebase
    - Creates DB record

    On failure at any step, the file is skipped and the error is logged and returned.

    Returns a summary of upload results per file.
    """
    if 'file' not in request.files:
        return JSONRes.new(400, "No file part in request.")

    files = request.files.getlist('file')
    if not files or all(file.filename == '' for file in files):
        return JSONRes.new(400, "No files selected.")

    fileSaveUpdates = {}

    for file in files:
        if file and file.filename: 
            # check file type
            if not allowed_file(file.filename):
                fileSaveUpdates[file.filename] = "ERROR: File type not allowed."
                continue

            # check file size
            file.seek(0, os.SEEK_END)
            file_length = file.tell()
            file.seek(0)  # reset pointer for reading later

            if file_length > MAX_FILE_SIZE:
                fileSaveUpdates[file.filename] = "ERROR: File too large. Max allowed is 5MB."
                continue
           
            # save file locally
            filename = Universal.generateUniqueID()
            filename = "{}.{}".format(filename, FileOps.getFileExtension(file.filename))
            fileObj = File(filename, 'artefacts')
    
            try:
                file.save(fileObj.path())
            except Exception as e:
                fileSaveUpdates[file.filename] = "ERROR: Failed to save locally. ({})".format(str(e))
                continue

            # sync local to firebase 
            res = FileManager.save(file=fileObj)
            if isinstance(res, str): # if fail
                Logger.log("DATAIMPORT UPLOAD FILEMANAGER ERROR: Failed to save file to FileManager: {}. ".format(res))
                
                remove = FileManager.removeLocally(fileObj) 
                if remove != True:
                    Logger.log("DATAIMPORT UPLOAD FILEMANAGER ERROR: Failed to   file locally: {}".format(remove))
                
                fileSaveUpdates[file.filename] = "ERROR: Failed to upload to FileManager: {}".format(res)
                continue

            # create artefact
            artefact = Artefact.fromFMFile(fileObj)
            
            try:
                artefact.save()
                fileSaveUpdates[file.filename] = "Uploaded successfully to database."

            except Exception as e:
                Logger.log("DATAIMPORT UPLOAD ARTEFACT ERROR: Failed to save artefact for {}: {}".format(file.filename, str(e)))

                FileManager.delete(file=fileObj)

                fileSaveUpdates[file.filename] = "ERROR: Failed to save artefact: {}".format(str(e))
                continue

    return JSONRes.new(200, "Upload results.", ResType.SUCCESS, updates=fileSaveUpdates)

    # @checkSession(strict=True, provideUser=True)
    # accID = session.get('accID') 
    # if not accID: 
    #     return JSONRes.unauthorised()
    
    # batch = Batch(user.id, batchArtefacts=savedArtefacts)
    # batch.initJob()
    # batch.save()

    # # Start async batch processing
    # batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
    # batch.job.save()
    
    # return JSONRes.new(200, "Files have sucessfully saved into database.")
    # raw={"batchID": batch.id}

@dataimportBP.route('/batches', methods=['GET'])
@checkSession(strict=True)
def getAllBatches():
    try:
        batches = Batch.load()  # Assuming returns list of all batches
        if not batches:
            return JSONRes.new(200, "No batches found.", raw={})

        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "All batches retrieved.", raw=formatted)

    except Exception as e:
        return JSONRes.ambiguousError(str(e))

@dataimportBP.route('/userbatches', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def getUserBatches(user: User):
    try:
        batches = Batch.load(userID=user.id, withArtefacts=False)
        if not batches:
            return JSONRes.new(200, "No batches found.", raw={})

        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "User batches retrieved.", raw=formatted)

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
            return JSONRes.new(200, "Batch job already completed")
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
    try:
        batch = Batch.load(id=batchID)
        if not batch:
            return JSONRes.new(404, "Batch not found.")

        if batch.userID != user.id:
            return JSONRes.unauthorised()

        batch.destroy()

        return JSONRes.new(200, "Batch {} and artefacts deleted.".format(batchID))

    except Exception as e:
        return JSONRes.ambiguousError(str(e))