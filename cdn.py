from flask import Blueprint, send_file, make_response, redirect
from fm import FileManager, File
from utils import JSONRes, ResType
import mimetypes, datetime
from services import Logger
from models import Category, Book
from decorators import checkSession

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

@cdnBP.route('/artefacts/<filename>')
@checkSession(strict=True)
def getArtefactImage(filename):
    """
    Serve an artefact image from the 'artefacts' store.
    """
    file = File(filename, "artefacts")

    fileExists = file.exists()
    if isinstance(fileExists, str):
        Logger.log(f"CDN GETARTEFACT ERROR: Failed to check file existence; response: {fileExists}")
        return JSONRes.new(500, ResType.ERROR, "Something went wrong.")

    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return 404
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log(f"CDN GETARTEFACT ERROR: {e}")
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
        Logger.log(f"CDN GETFACE ERROR: Failed to check file existence; response: {fileExists}")
        return JSONRes.new(500, ResType.ERROR, "Something went wrong.")

    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return 404
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log(f"CDN GETFACE ERROR: {e}")
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
        Logger.log(f"CDN GETASSET ERROR: Failed to check file existence; response: {fileExists}")
        return JSONRes.new(500, ResType.ERROR, "Something went wrong.")

    if not fileExists[0]:
        return JSONRes.new(404, ResType.ERROR, "Requested file not found.")

    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            return 404
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log(f"CDN GETASSET ERROR: {e}")
        return JSONRes.new(500, ResType.ERROR, "Error sending file.")

@cdnBP.route('/getCatalogue')
@checkSession(strict=True)
def getAllCategoriesWithArtefacts():
    try:
        categories = Category.load()
        if not categories:
            return JSONRes.new(404, ResType.ERROR, "No categories available.")

        result = {}

        for cat in categories:
            if not cat.members:
                continue

            artefactsList = []
            for catArt in cat.members.values():
                if not catArt.getArtefact() or not catArt.artefact:
                    continue

                artefact = catArt.artefact
                file_check = FileManager.prepFile("artefacts", artefact.image)
                if isinstance(file_check, str):
                    Logger.log(f"CDN GETCATALOGUE ERROR: Failed to prepare file '{artefact.image}'; response: {file_check}")
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

        books = Book.load() or []
        bookList = [
            {
                "id": book.id,
                "title": book.title,
                "subtitle": book.subtitle,
                "mmIDs": book.mmIDs
            }
            for book in books
        ]

        return JSONRes.new(200, ResType.SUCCESS, data={
            "categories": result,
            "books": bookList
        })

    except Exception as e:
        Logger.log(f"CDN GETALL EXCEPTION: {e}")
        return JSONRes.new(500, ResType.ERROR, "Unexpected ERROR occurred.")
