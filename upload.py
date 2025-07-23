import os, time
from werkzeug.utils import secure_filename
from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from services import Logger, Encryption, Universal, FileOps
from sessionManagement import checkSession
from models import Metadata, Artefact, User, Batch, BatchArtefact, BatchProcessingJob
from fm import FileManager
from services import ThreadManager
from metagen import MetadataGenerator
from ingestion import DataImportProcessor

dataimportBP = Blueprint('dataimport', __name__, url_prefix="/dataimport")

@dataimportBP.route('/upload', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def upload(user: User):
    accID = session.get('accID')
    if not accID:
        return JSONRes.unauthorised()

    if 'file' not in request.files:
        return JSONRes.new(400, "No file part in request.")

    files = request.files.getlist('file')
    if not files or all(file.filename == '' for file in files):
        return JSONRes.new(400, "No files selected.")

    savedArtefacts = []

    for file in files:
        if file and file.filename:
            filename = Universal.generateUniqueID()
            ext = FileOps.getFileExtension(file.filename)
            fullFilename = "{}.{}".format(filename, ext)
            imagePath = os.path.join("artefacts", fullFilename)
            file.save(imagePath)

            artefact = Artefact(filename, imagePath, metadata=None)
            artefact.save()
            savedArtefacts.append(artefact)

    FileManager.save()

    batch = Batch(user.id, batchArtefacts=savedArtefacts)
    batch.save()

    job = BatchProcessingJob(batch=batch, jobID=None, status=BatchProcessingJob.Status.PENDING)
    job.save()

    batch.job = job
    batch.save()

    # Start async batch processing
    ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)


    return JSONRes.new(200, "Batch is processing.", raw={"batchID": batch.id})

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