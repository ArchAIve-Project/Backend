import os, time
from werkzeug.utils import secure_filename
from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from services import Logger, Encryption, Universal, FileOps
from sessionManagement import checkSession
from models import Metadata, Artefact, User, Batch, BatchArtefact, BatchProcessingJob
from fm import FileManager, File, FileUtils
from services import ThreadManager
from metagen import MetadataGenerator
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

    for file in files:
        if file and file.filename: 
           # Check file type
            if not FileOps.allowedFileExtension(file.filename):
                fileSaveUpdates[file.filename] = "ERROR: File type not allowed."
                continue

            # Check file size
            validSize = FileOps.fileSize(file)
            if not validSize:
                fileSaveUpdates[file.filename] = "ERROR: File size > 10 MB."
                continue
           
            # save file locally
            filename = Universal.generateUniqueID()
            filename = "{}.{}".format(filename, FileOps.getFileExtension(file.filename))
            fileObj = File(filename, 'artefacts')
    
            try:
                file.save(fileObj.path())
            except Exception as e:
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save file: {}".format(e))
                fileSaveUpdates[file.filename] = "ERROR: Failed to save file: {}".format(str(e))
                continue

            # sync local to firebase 
            res = FileManager.save(file=fileObj)
            if isinstance(res, str): # if fail
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save file: {}. ".format(res))
                
                remove = FileManager.removeLocally(fileObj) 
                if remove != True:
                    Logger.log("DATAIMPORT UPLOAD ERROR: Failed to remove file: {}".format(remove))
                
                fileSaveUpdates[file.filename] = "ERROR: Failed to save file: {}".format(res)
                continue

            # create artefact
            artefact = Artefact.fromFMFile(fileObj)
            
            try:
                artefact.save()
                sucessfulFiles.append(artefact)
                fileSaveUpdates[file.filename] = "File uploaded successfully."

            except Exception as e:
                Logger.log("DATAIMPORT UPLOAD ERROR: Failed to save artefact for {}: {}".format(file.filename, str(e)))

                FileManager.delete(file=fileObj)

                fileSaveUpdates[file.filename] = "ERROR: Failed to save artefact: {}".format(str(e))
                continue
    
    if not sucessfulFiles:
        return JSONRes.new(400, "No artefacts were successfully uploaded.", ResType.ERROR, updates=fileSaveUpdates)
    
    try:
        batch = Batch(user.id, batchArtefacts=sucessfulFiles)
        batch.save()

    except Exception as e:
        Logger.log("DATAIMPORT UPLOAD ERROR: Failed to create batch: {}".format(e))

    return JSONRes.new(200, "Upload results.", ResType.SUCCESS, updates=fileSaveUpdates, batchID=batch.id)

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