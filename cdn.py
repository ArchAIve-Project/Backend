from flask import Blueprint, send_file, make_response, redirect
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes, datetime
from services import Logger
from models import Category, Book, Artefact
from decorators import checkSession, cache

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

@cdnBP.route('/artefacts/<artefactId>')
@checkSession(strict=True)
def getArtefactImage(artefactId):
    """
    Serve an artefact image from the 'artefacts' store using the artefact ID.
    """
    try:
        # Load artefact by ID
        art = Artefact.load(id=artefactId, includeMetadata=False)
        if not art:
            return JSONRes.new(404, ResType.ERROR, "Artefact not found.")

        # Retrieve the filename from the artefact
        filename = art.image
        if not filename:
            return JSONRes.new(404, ResType.ERROR, "Artefact has no associated image.")

        # Create File object using the filename and store
        file = File(filename, "artefacts")

        # Check if file exists in cloud
        fileExists = file.exists()
        if isinstance(fileExists, str):
            Logger.log("CDN GETARTEFACT ERROR: Failed to check file existence; response: {}".format(fileExists))
            return JSONRes.ambiguousError()

        if not fileExists[0]:
            return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

        # Generate signed URL and redirect
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        if url.startswith("ERROR"):
            return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

        res = make_response(redirect(url))

        # Set headers to prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"

        return res

    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: {}".format(e))
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
        Logger.log("CDN: Failed to load artefacts - {}".format(e))

    # Load all categories (with artefact references inside each category)
    try:
        categories = Category.load(withArtefacts=True) or []
    except Exception as e:
        Logger.log("CDN: Failed to load categories - {}".format(e))

    result = {}

    # Loop through each category and compile its artefact list
    for cat in categories:
        if not cat.members:
            continue  # Skip empty categories

        artefactsList = []
        for artefact_id, catArt in cat.members.items():
            try:
                art = artefactMap.get(artefact_id)  # Look up the artefact from preloaded map
                if art:
                    artefactsList.append({
                        "id": art.id,
                        "name": art.name,
                        "image": art.image,
                        "description": art.description
                    })
            except Exception as e:
                Logger.log("CDN: Failed to resolve artefact in category {} - {}".format(cat.name, e))

        if artefactsList:
            result[cat.name] = artefactsList  # Add category to result if it has artefacts

    # Load all books (which reference artefact IDs)
    try:
        books = Book.load() or []
    except Exception as e:
        Logger.log("CDN: Failed to load books - {}".format(e))

    bookList = []

    # Loop through each book to build its associated artefacts
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
                else:
                    # Artefact not found — fallback to just ID
                    mmDetails.append({
                        "id": mmID,
                        "name": mmID,
                        "description": ""
                    })
            except Exception as e:
                Logger.log("CDN: Failed to resolve mmID {} in book {} - {}".format(mmID, book.id, e))

        bookList.append({
            "id": book.id,
            "title": book.title,
            "subtitle": book.subtitle,
            "mmArtefacts": mmDetails  # Each book now includes resolved artefacts
        })

    # Return structured catalogue of categories and books
    return JSONRes.new(200, ResType.SUCCESS, data={
        "categories": result,
        "books": bookList
    })

