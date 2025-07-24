from flask import Blueprint, send_file, make_response, redirect
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes, datetime
from services import Logger
from models import Category, Book, Artefact
from decorators import checkSession, cache, timeit

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

@cdnBP.route('/artefacts/<artefactId>')
@checkSession(strict=True)
def getArtefactImage(artefactId):
    """
    Serve an artefact image from the 'artefacts' store using the artefact ID.
    """
    # Attempt to load the artefact
    try:
        art = Artefact.load(id=artefactId, includeMetadata=False)
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to load artefact '{}': {}".format(artefactId, e))
        return JSONRes.new(500, ResType.ERROR, "Error loading artefact.")

    if not art:
        return JSONRes.new(404, ResType.ERROR, "Artefact not found.")

    # Attempt to get image filename
    filename = art.image
    if not filename:
        return JSONRes.new(404, ResType.ERROR, "Artefact has no associated image.")

    # Attempt to create file object
    try:
        file = File(filename, "artefacts")
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to construct File object '{}': {}".format(filename, e))
        return JSONRes.ambiguousError()

    # Attempt to check if file exists in the cloud
    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log("CDN GETARTEFACT ERROR: File existence check returned error string: {}".format(fileExists))
        return JSONRes.ambiguousError()
    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    # Attempt to generate signed URL
    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to generate signed URL for '{}': {}".format(filename, e))
        return JSONRes.new(500, ResType.ERROR, "Error generating signed URL.")

    if url.startswith("ERROR"):
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    # Attempt to redirect with no-cache headers
    try:
        res = make_response(redirect(url))
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        return res
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to construct response for artefact '{}': {}".format(artefactId, e))
        return JSONRes.new(500, ResType.ERROR, "Error sending file.")

@cdnBP.route('/people/<filename>')
@checkSession(strict=True)
def getFaceImage(filename):
    """
    Serve a face image from the 'people' store.
    """
    file = File(filename, "people")

    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log("CDN GETFACE ERROR: Failed to check file existence; response: {}".format(fileExists))
        return JSONRes.ambiguousError()

    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return JSONRes.new(404, ResType.ERROR, "Requested file not found.")
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log("CDN GETFACE ERROR: {}".format(e))
        return JSONRes.new(500, ResType.ERROR, "Error sending file.")

@cdnBP.route('/asset/<filename>')
@checkSession(strict=True)
def getAsset(filename):
    """
    Serve a public file from the 'FileStore' store (via /FileStore).
    """    
    file = File(filename, "FileStore")

    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log("CDN GETASSET ERROR: Failed to check file existence; response: {}".format(fileExists))
        return JSONRes.ambiguousError()

    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return JSONRes.new(404, ResType.ERROR, "Requested file not found.")
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log("CDN GETASSET ERROR: {}".format(e))
        return JSONRes.new(500, ResType.ERROR, "Error sending file.")

@cdnBP.route('/catalogue')
@checkSession(strict=True)
@timeit
@cache
def getAllCategoriesWithArtefacts():
    """
    Returns all artefacts grouped by category and a list of books with their associated artefacts.

    This endpoint:
    - Loads all artefacts into a lookup map.
    - Loads all categories and links each category to its artefacts using the map.
    - Loads all books and links each book to its listed artefact IDs (mmIDs).
    - Handles exceptions at a granular level (load calls and lookup failures).
    - Caches the output and requires session authentication.
    
    Response Format:
    ```
    {
        "categories": {
            "CategoryName1": [
                { "id": ..., "name": ..., "image": ..., "description": ... },
                ...
            ],
            ...
        },
        "books": [
            {
                "id": ...,
                "title": ...,
                "subtitle": ...,
                "mmArtefacts": [
                    { "id": ..., "name": ..., "description": ... },
                    ...
                ]
            },
            ...
        ]
    }
    """

    artefactMap = {}
    categories = []
    books = []

    # Load all artefacts into a dictionary for quick lookup by ID
    try:
        all_artefacts = Artefact.load() or []
        artefactMap = {art.id: art for art in all_artefacts}
    except Exception as e:
        Logger.log(f"CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load artefacts - {e}")
        return JSONRes.new(500, ResType.ERROR, "Failed to load artefacts.")

    # Load all categories
    try:
        categories = Category.load() or []
    except Exception as e:
        Logger.log(f"CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load categories - {e}")
        return JSONRes.new(500, ResType.ERROR, "Failed to load categories.")

    result = {}

    # Build category -> artefact mapping
    for cat in categories:
        artefactsList = []
        if cat.members:
            for artefact_id, catArt in cat.members.items():
                try:
                    art = artefactMap.get(artefact_id)
                    if art:
                        artefactsList.append({
                            "id": art.id,
                            "name": art.name,
                            "image": art.image,
                            "description": art.description
                        })
                except Exception as e:
                    Logger.log(f"CDN: Failed to resolve artefact in category {cat.name} - {e}")
        result[cat.name] = artefactsList

    # Load all books
    try:
        books = Book.load() or []
    except Exception as e:
        Logger.log(f"CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load books - {e}")
        return JSONRes.new(500, ResType.ERROR, "Failed to load books.")

    bookList = []

    for book in books:
        mmDetails = []
        for mmID in book.mmIDs:
            try:
                art = artefactMap.get(mmID)
                if art:
                    mmDetails.append({
                        "id": art.id,
                        "name": art.name,
                        "description": art.description
                    })
            except Exception as e:
                Logger.log(f"CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to resolve mmID {mmID} in book {book.id} - {e}")
        bookList.append({
            "id": book.id,
            "title": book.title,
            "subtitle": book.subtitle,
            "mmArtefacts": mmDetails
        })

    return JSONRes.new(200, ResType.SUCCESS, data={
        "categories": result,
        "books": bookList
    })
