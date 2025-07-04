from flask import Blueprint, send_file, make_response
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes
from services import Logger
import os

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

@cdnBP.route('/artefact/<filename>')
def getArtefactImage(filename):
    """
    Serve an artefact image from the 'artefacts' store.
    """
    FileManager.setup()
    fileObj = FileManager.prepFile("artefact", filename)

    if not isinstance(fileObj, File):
        Logger.log(f"CDN GETARTEFACT ERROR: {fileObj}")
        if str(fileObj).strip().lower() == "error: file does not exist.":
            return JSONRes.new(404, ResType.error, "Requested file not found.")
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    localPath = fileObj.path()
    ext = os.path.splitext(localPath)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return JSONRes.new(400, ResType.error, f"Invalid image extension '{ext}'")

    mime_type = mimetypes.guess_type(localPath)[0] or 'application/octet-stream'
    try:
        response = make_response(send_file(localPath, mimetype=mime_type))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception as e:
        Logger.log(f"CDN GETARTEFACT ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")

@cdnBP.route('/people/<filename>')
def getFaceImage(filename):
    """
    Serve a face image from the 'people' store.
    """
    FileManager.setup()
    fileObj = FileManager.prepFile("people", filename)

    if not isinstance(fileObj, File):
        Logger.log(f"CDN GETFACE ERROR: {fileObj}")
        if str(fileObj).strip().lower() == "error: file does not exist.":
            return JSONRes.new(404, ResType.error, "Requested file not found.")
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    localPath = fileObj.path()
    ext = os.path.splitext(localPath)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return JSONRes.new(400, ResType.error, f"Invalid image extension '{ext}'")

    mime_type = mimetypes.guess_type(localPath)[0] or 'application/octet-stream'
    try:
        response = make_response(send_file(localPath, mimetype=mime_type))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception as e:
        Logger.log(f"CDN GETFACE ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")

@cdnBP.route('/FileStore/<filename>')
def getAsset(filename):
    """
    Serve a public file from the 'FileStore' store (via /FileStore).
    """
    FileManager.setup()
    store = "FileStore"

    fileObj = FileManager.prepFile(store, filename)
    if not isinstance(fileObj, File):
        Logger.log(f"CDN GETASSET ERROR: {fileObj}")
        if str(fileObj).strip().lower() == "error: file does not exist.":
            return JSONRes.new(404, ResType.error, "Requested file not found.")
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    localPath = fileObj.path()
    ext = os.path.splitext(localPath)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return JSONRes.new(400, ResType.error, f"Invalid image extension '{ext}'")

    mime_type = mimetypes.guess_type(localPath)[0] or 'application/octet-stream'
    try:
        response = make_response(send_file(localPath, mimetype=mime_type))
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response
    except Exception as e:
        Logger.log(f"CDN GETASSET ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")

