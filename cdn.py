from flask import Blueprint, send_file, make_response
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes
from services import Logger
import os

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

@cdnBP.route('/image/<store>/<filename>')
def getImage(store, filename):
    """
    Serve an image file from the specified storage 'store' and 'filename'.

    Workflow:
    - Initialize FileManager.
    - Retrieve the file object for the given store and filename.
    - Confirm the file exists locally and has an allowed image extension.
    - Determine the MIME type of the file.
    - Send the image file with appropriate HTTP headers.
    - Return JSON-formatted error responses on failure.

    Args:
        store (str): The storage category or folder name.
        filename (str): The name of the image file to retrieve.

    Returns:
        On success:
            - Flask Response streaming the image file with correct Content-Type
            - and caching headers to prevent stale content.

        On error:
            JSON response with keys:
            - type: "ERROR"
            - message: Description of the error
            - HTTP status codes such as 400 (bad extension), 404 (file missing), or 500 (server error)

    ## Frontend Usage:
        Use a HTTP client (like your custom 'server' instance) to request the image
        with responseType set to 'blob'. For example:

        ```js
        server.get('<backendBaseURL>/cdn/image/<store>/<imageName>', {
            responseType: 'blob',
        }).then(res => {
            const blob = res.data?.data;
            if (!(blob instanceof Blob)) {
                // Handle error: response is not a blob
            }
            if (!blob.type.startsWith('image/')) {
                // Handle error: unexpected MIME type
            }
            const imageUrl = URL.createObjectURL(blob);
            // Use imageUrl as src in <img> or <Image> components
        }).catch(err => {
            // Handle errors appropriately
        });
        ```
    """
    # Extract the file extension and validate against allowed image extensions
    ext = os.path.splitext(localPath)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return JSONRes.new(400, ResType.error, f"Invalid image extension '{ext}'")
    
    FileManager.setup()
    fileObj = FileManager.prepFile(store, filename)

    # If retrieval failed, return JSON error response with 404 status
    if not isinstance(fileObj, File):
        Logger.log(f"CDN GETIMAGE ERROR: {fileObj}")
        # Return 404 only if it's the specific "file not found" case
        if str(fileObj).strip().lower() == "error: file does not exist.":
            return JSONRes.new(404, ResType.error, "Requested file not found.")

        return JSONRes.new(500, ResType.error, "Failed to retrieve file.")

    # Get the local filesystem path of the retrieved file object
    localPath = fileObj.path()

    # Guess the MIME type based on the file extension; fallback to generic binary
    mime_type = mimetypes.guess_type(localPath)[0] or 'application/octet-stream'

    try:
        # Prepare a Flask response
        response = make_response(send_file(localPath, mimetype=mime_type))

        # Set headers to disable caching so clients always get the latest file
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'

        return response
    except Exception as e:
        Logger.log(f"CDN GETIMAGE ERROR: {e}")
        return JSONRes.new(500, ResType.error, f"Error sending file.")

@cdnBP.route('/artefact/<filename>')
def getArtefactImage(filename):
    """
    Serve an artefact image from the 'artefacts' store.
    """
    FileManager.setup()
    fileObj = FileManager.prepFile("artefacts", filename)

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

@cdnBP.route('/face/<filename>')
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

@cdnBP.route('/asset/<store>/<filename>')
def getAsset(store, filename):
    """
    Serve a generic image asset from a whitelisted store.
    """
    FileManager.setup()
    allowedStores = {"artefacts", "people", "public"}

    if store not in allowedStores:
        return JSONRes.new(403, ResType.error, "Access to this store is forbidden.")

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
