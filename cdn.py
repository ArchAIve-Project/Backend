from flask import Blueprint, send_file, make_response, redirect
from fm import FileManager, File
from utils import JSONRes, ResType
import datetime
from typing import Dict, List
from services import Logger
from models import Category, Book, Artefact
from decorators import cache, timeit
from sessionManagement import checkSession
import json

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
        return JSONRes.ambiguousError()

    if not art:
        return JSONRes.new(404, "Artefact not found.")

    # Attempt to get image filename
    filename = art.image
    if not filename:
        return JSONRes.new(404, "Artefact has no associated image.")

    # Attempt to create file object
    file = File(filename, "artefacts")

    # Attempt to generate signed URL
    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        if url.startswith("ERROR"):
            if url == "ERROR: File does not exist.":
                return JSONRes.new(404, "Requested file not found.")
            else:
                raise Exception(url)
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to generate signed URL for '{}': {}".format(artefactId, e)) # changed to artefactId
        return JSONRes.ambiguousError()

    # Attempt to redirect with no-cache headers
    try:
        res = make_response(redirect(url))
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        return res
    except Exception as e:
        Logger.log("CDN GETARTEFACT ERROR: Failed to construct response for artefact '{}': {}".format(artefactId, e))
        return JSONRes.ambiguousError()

@cdnBP.route('/people/<filename>')
@checkSession(strict=True)
def getFaceImage(filename):
    """
    Serve a face image from the 'people' store.
    """
    file = File(filename, "people")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return JSONRes.new(404, "Requested file not found.")
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log(f"CDN GETFACE ERROR: {e}")
        return JSONRes.new(500, "Error sending file.")

@cdnBP.route('/asset/<filename>')
@checkSession(strict=True)
def getAsset(filename):
    """
    Serve a public file from the 'FileStore' store (via /FileStore).
    """    
    file = File(filename, "FileStore")
    
    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return JSONRes.new(404, "Requested file not found.")
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log("CDN GETASSET ERROR: {}".format(e))
        return JSONRes.ambiguousError()

@cdnBP.route('/catalogue')
@checkSession(strict=True)
@cache
def getAllCategoriesWithArtefacts():
    """
    Returns all human-figure artefacts grouped by category and all books with their associated
    meeting-minute artefacts.

    This endpoint:
    - Loads all artefacts (without metadata) into a lookup map, with `artType` used to differentiate types.
    - Loads all categories and links each to its human-figure artefacts (artType = 'hf').
    - Loads all books and links each to its meeting-minute artefacts (artType = 'mm') via mmIDs.
    - Handles exceptions at each loading or lookup stage.
    - Caches the output and requires session authentication.

    Response Format:
    ```json
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
        all_artefacts = Artefact.load(includeMetadata=False) or []
        artefactMap: Dict[str, Artefact] = {art.id: art for art in all_artefacts}
    except Exception as e:
        Logger.log("CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load artefacts - {}".format(e))
        return JSONRes.ambiguousError()

    # Load all categories
    try:
        categories: List[Category] = Category.load() or []
    except Exception as e:
        Logger.log("CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load categories - {}".format(e))
        return JSONRes.ambiguousError()

    result = {}

    for cat in categories:
        artefactsList = []
        if cat.members:
            for artefact_id, _ in cat.members.items():
                try:
                    art = artefactMap.get(artefact_id)
                    if art and art.artType == "hf":
                        artefactsList.append({
                            "id": art.id,
                            "name": art.name,
                            "image": art.image,
                            "description": art.description
                        })
                except Exception as e:
                    Logger.log("CDN: Failed to resolve artefact in category {} - {}".format(cat.name, e))
        result[cat.name] = artefactsList

    # Load all books
    try:
        books: List[Book] = Book.load() or []
    except Exception as e:
        Logger.log("CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to load books - {}".format(e))
        return JSONRes.ambiguousError()

    bookList = []

    for book in books:
        mmDetails = []
        for mmID in book.mmIDs:
            try:
                art = artefactMap.get(mmID)
                if art and art.artType == "mm":
                    mmDetails.append({
                        "id": art.id,
                        "name": art.name,
                        "description": art.description
                    })
            except Exception as e:
                Logger.log("CDN GETALLCATEGORIESWITHARTEFACTS ERROR: Failed to resolve mmID {} in book {} - {}".format(mmID, book.id, e))
        bookList.append({
            "id": book.id,
            "title": book.title,
            "subtitle": book.subtitle,
            "mmArtefacts": mmDetails
        })

    return JSONRes.new(200, "Retrieval success.", data={
        "categories": result,
        "books": bookList
    })


@cdnBP.route('/artefactMetadata/<artID>')
@checkSession(strict=True)
@cache
def getArtefactMetedata(artID):
    """
    Returns the metadata of an artefact.
    """
    try:
        art = Artefact.load(id=artID, includeMetadata=True)
        if not isinstance(art, Artefact):
            return JSONRes.new(404, "Artefact not found.")
        
        return JSONRes.new(200, "Retrieval success.", data=art.metadata.represent())

    except Exception as e:
        Logger.log("CDN GETARTEFACTMETADATA ERROR: Failed to load artefact metadata - {}".format(e))
        return JSONRes.ambiguousError()