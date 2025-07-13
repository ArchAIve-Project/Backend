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

uploadBP = Blueprint('upload', __name__, url_prefix="/upload")

UPLOAD_FOLDER = "uploads"  # check if folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@uploadBP.route('/', methods=['POST'])
@checkSession(strict=True, provideUser=True)
def upload(user: User):
    accID = session.get('accID')
    if not accID:
        return JSONRes.new(401, "ACCOUNTID FAILED ERROR: Failed to retrieve")
    
    print("Request content type:", request.content_type)
    print("Request files:", request.files)
    print("Request form:", request.form)

    if 'file' not in request.files:
        return JSONRes.new(400, "No file part in the request.")

    files = request.files.getlist('file')
    if not files or all(file.filename == '' for file in files):
        return JSONRes.new(400, "No files selected.")

    Logger.log(f"Received files: {[f.filename for f in files]}")
    savedArtefacts = []

    for file in files:
        if file and file.filename != '':
            filename = Universal.generateUniqueID()
            ext = FileOps.getFileExtension(file.filename)
            fullFilename = f"{filename}.{ext}"
            imagePath = os.path.join(UPLOAD_FOLDER, fullFilename)
            file.save(imagePath)

            artefact = Artefact("testing", imagePath, metadata=None)
            artefact.save()
            savedArtefacts.append(artefact)

    FileManager.save()

    # Create a batch with these artefacts as unprocessed
    batch = Batch(user.id, unprocessed=savedArtefacts, processed=None, confirmed=None)
    batch.save()

    # Start processing
    ThreadManager.defaultProcessor.addJob(ImportProcessor.processBatch, batch)

    return JSONRes.new(200, "Batch is processing.")

class ImportProcessor:
    @staticmethod
    def processBatch(batch: Batch, chunk_size=10):
        batch.getArtefacts(unprocessed=True)
        unprocessedItems = [a for a in batch.unprocessed.values() if isinstance(a, Artefact)]
        total = len(unprocessedItems)
        i = 0

        while i < total:
            currentChunk = unprocessedItems[i:i + chunk_size]
            # Submit one job per chunk to the default processor
            ThreadManager.defaultProcessor.addJob(ImportProcessor.processChunk, currentChunk, batch)

            # Wait for this chunk to be processed before moving on
            while not all(artefact.id in batch.processed for artefact in currentChunk):
                time.sleep(0.5)

            i += chunk_size

        batch.save()
        return True


    @staticmethod
    def processChunk(chunk, batch):
        for artefact in chunk:
            ImportProcessor.processItem(artefact)
            batch.move(artefact, Batch.Stage.PROCESSED)

    @staticmethod
    def processItem(artefact: Artefact):
        out = MetadataGenerator.generate(artefact.image)
        artefact.metadata = Metadata.fromMetagen(artefact.id, out)
        artefact.save()
        return artefact
        