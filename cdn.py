from flask import Blueprint, send_file, make_response
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes
from services import Logger
from models import Category, Book

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

@cdnBP.route('/getCatalogue')
def getAllCategoriesWithArtefacts():
    try:
        categories = Category.load()
        if not categories or not isinstance(categories, list):
            return JSONRes.new(404, ResType.ERROR, "No categories available.")

        result = {}

        for cat in categories:
            if not cat.members:
                continue

            artefactsList = []
            for catArt in cat.members.values():
                loaded = catArt.getArtefact()
                if not loaded:
                    continue

                artefact = catArt.artefact
                if not artefact:
                    continue

                file = FileManager.prepFile("artefacts", artefact.image)
                if isinstance(file, str):
                    Logger.log(f"CDN GETCATALOGUE ERROR: Failed to prepare file '{artefact.image}'; response: {file}")
                    continue

                artefactsList.append({
                    "id": artefact.id,
                    "name": artefact.name,
                    "image": artefact.image,
                    "description": artefact.description
                })

            if artefactsList:
                result[cat.name] = artefactsList

        if not result:
            return JSONRes.new(404, ResType.ERROR, "No artefacts found in any category.")

        books = Book.load()
        bookList = []
        if books and isinstance(books, list):
            for book in books:
                bookList.append({
                    "id": book.id,
                    "title": book.title,
                    "subtitle": book.subtitle,
                    "mmIDs": book.mmIDs
                })

        return JSONRes.new(200, ResType.SUCCESS, data={
            "categories": result,
            "books": bookList
        })

    except Exception as e:
        Logger.log(f"CDN GETALL EXCEPTION: {e}")
        return JSONRes.new(500, ResType.ERROR, "Unexpected error occurred.")

