from flask import Blueprint, send_file, make_response
from fm import FileManager
from utils import JSONRes, ResType
import mimetypes
import os

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

@cdnBP.route('/image/<store>/<filename>')
def get_image(store, filename):
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
        server.get('<backendBaseURL>/cdn/image/people/face_693eb27d9de94bc59102ff7756b5073a.jpg', {
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
    
    FileManager.setup()
    file_obj = FileManager.prepFile(store, filename)

    # If retrieval failed, return JSON error response with 404 status
    if isinstance(file_obj, str) and file_obj.startswith("ERROR"):
        return JSONRes.new(404, ResType.error, file_obj)

    # Get the local filesystem path of the retrieved file object
    local_path = file_obj.path()

    # Verify that the file actually exists locally; else return 404 error
    if not os.path.isfile(local_path):
        return JSONRes.new(404, ResType.error, "File does not exist locally")

    # Extract the file extension and validate against allowed image extensions
    ext = os.path.splitext(local_path)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return JSONRes.new(400, ResType.error, f"Invalid image extension '{ext}'")

    # Guess the MIME type based on the file extension; fallback to generic binary
    mime_type = mimetypes.guess_type(local_path)[0] or 'application/octet-stream'

    try:
        # Prepare a Flask response
        response = make_response(send_file(local_path, mimetype=mime_type))

        # Set headers to disable caching so clients always get the latest file
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'

        return response
    except Exception as e:
        return JSONRes.new(500, ResType.error, f"Error sending file: {e}")
