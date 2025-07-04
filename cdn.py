from flask import Blueprint, send_file, make_response
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes
from services import Logger

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

@cdnBP.route('/artefacts/<filename>')
def getArtefactImage(filename):
    """
    Serve an artefact image from the 'artefacts' store.
    """
    file = File(filename, "artefacts")
    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log(f"CDN GETARTEFACT ERROR: Failed to check file existence; response: {fileExists}")
        return JSONRes.new(500, ResType.error, "Something went wrong.")

    if not fileExists[0]:
        return JSONRes.new(404, ResType.error, "Requested file not found.")

    fileBytes = file.getBytes()
    if isinstance(fileBytes, str):
        Logger.log(f"CDN GETARTEFACT ERROR: Failed to retrieve file bytes; response: {fileBytes}")
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    try:
        response = make_response(fileBytes)
        response.headers.set('Content-Type', mime_type)
        response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
        return response
    except Exception as e:
        Logger.log(f"CDN GETARTEFACT ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")


@cdnBP.route('/people/<filename>')
def getFaceImage(filename):
    """
    Serve a face image from the 'people' store.
    """
    file = File(filename, "people")
    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log(f"CDN GETFACE ERROR: Failed to check file existence; response: {fileExists}")
        return JSONRes.new(500, ResType.error, "Something went wrong.")

    if not fileExists[0]:
        return JSONRes.new(404, ResType.error, "Requested file not found.")

    fileBytes = file.getBytes()
    if isinstance(fileBytes, str):
        Logger.log(f"CDN GETFACE ERROR: Failed to retrieve file bytes; response: {fileBytes}")
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    try:
        response = make_response(fileBytes)
        response.headers.set('Content-Type', mime_type)
        response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
        return response
    except Exception as e:
        Logger.log(f"CDN GETFACE ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")

@cdnBP.route('/asset/<filename>')
def getAsset(filename):
    """
    Serve a public file from the 'FileStore' store (via /FileStore).
    """    
    file = File(filename, "FileStore")
    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log("CDN GETASSET ERROR: Failed to check file existence; response: {}".format(fileExists))
        return JSONRes.new(500, ResType.error, "Something went wrong.")
    
    if not fileExists[0]:
        return JSONRes.new(404, ResType.error, "Requested file not found.")
    
    fileBytes = file.getBytes()
    if isinstance(fileBytes, str):
        Logger.log("CDN GETASSET ERROR: Failed to retrieve file bytes; response: {}".format(fileBytes))
        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")
    
    mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

    try:
        response = make_response(fileBytes)
        response.headers.set('Content-Type', mime_type)
        response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
        return response
    except Exception as e:
        Logger.log(f"CDN GETASSET ERROR: {e}")
        return JSONRes.new(500, ResType.error, "Error sending file.")

