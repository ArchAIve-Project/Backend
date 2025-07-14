import os, time
from werkzeug.utils import secure_filename
from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from services import Logger, Encryption, Universal, FileOps
from decorators import jsonOnly, enforceSchema, checkSession
from models import Metadata, Batch, Artefact, User
from fm import FileManager
from services import ThreadManager
from metagen import MetadataGenerator
from importing import ImportProcessor

uploadBP = Blueprint('upload', __name__, url_prefix="/upload")

@uploadBP.route('/', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def upload(user: User):
    accID = session.get('accID')
    if not accID:
        return JSONRes.unauthorised()

    if 'file' not in request.files:
        return JSONRes.new(400, "No file part in the request.")

    files = request.files.getlist('file')
    if not files or all(file.filename == '' for file in files):
        return JSONRes.new(400, "No files selected.")

    savedArtefacts = []

    for file in files:
        if file and file.filename != '':
            filename = Universal.generateUniqueID()
            ext = FileOps.getFileExtension(file.filename)
            fullFilename = f"{filename}.{ext}"
            imagePath = os.path.join("artefacts", fullFilename)
            file.save(imagePath)

            artefact = Artefact("testing", imagePath, metadata=None)
            artefact.save()
            savedArtefacts.append(artefact)

    FileManager.save()

    batch = Batch(user.id, unprocessed=savedArtefacts, processed=None, confirmed=None)
    batch.save()

    # Logger

    ThreadManager.defaultProcessor.addJob(ImportProcessor.processBatch, batch)

    return JSONRes.new(200, "Batch is processing.")

@uploadBP.route('/batches', methods=['GET'])
@checkSession(strict=True)
def allBatches():
    try:
        batches = Batch.load()
        if not batches:
            return JSONRes.new(200, "No batches found.", raw={})
        
        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "All batches retrieved.", raw=formatted)
    
    except Exception as e: 
        Logger.log(f"BATCH LIST ERROR: {e}")
        return JSONRes.ambiguousError(str(e))


@uploadBP.route('/userbatches', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def userBatches(user: User):
    Logger.log(f"userBatches called for user id: {user.id}")
    try:
        batches = Batch.load(userID=user.id, withArtefacts=False)
        Logger.log(f"Loaded batches count: {len(batches) if batches else 0}")
        if not batches:
            return JSONRes.new(200, "No batches found.", raw={})
        
        formatted = {b.id: b.represent() for b in batches}
        return JSONRes.new(200, "Batches retrieved.", raw=formatted)
    
    except Exception as e:
        Logger.log(f"userBatches error: {e}")
        return JSONRes.ambiguousError(str(e))
    
@uploadBP.route('/batches/<batchID>', methods=['DELETE'])
@checkSession(strict=True, provideUser=True)
def deleteBatch(user: User, batchID: str):
    try:
        batch = Batch.load(id=batchID)
        if not batch:
            return JSONRes.new("Batch not found.")

        if batch.userID != user.id:
            return JSONRes.unauthorised()

        batch.getArtefacts(unprocessed=True, processed=True, confirmed=True)
        batch.destroy()

        return JSONRes.new(200, f"Batch {batchID} and artefacts deleted.")

    except Exception as e:
        Logger.log("DELETE BATCH ERROR: {}".format(e))
        return JSONRes.ambiguousError()
