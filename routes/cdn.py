import os, datetime
from fm import FileManager, File
from utils import JSONRes
from typing import Dict, List
from flask import Blueprint, send_file, make_response, redirect, request
from services import Logger
from schemas import Category, Book, Artefact, User, Batch
from decorators import cache
from sessionManagement import checkSession

cdnBP = Blueprint('cdn', __name__, url_prefix='/cdn')

@cdnBP.route('/profileInfo/<userID>', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def getProfileInfo(user: User, userID: str=None):
    # Carry out authorisation checks
    if userID != user.id:
        if not user.superuser:
            # User is not the same as the requested user and is not a superuser
            return JSONRes.unauthorised()
        else:
            # User is a superuser, so we can load the requested user
            try:
                user = User.load(id=userID)
                if user is None:
                    return JSONRes.new(404, "User not found.")
                if not isinstance(user, User):
                    raise Exception("Unexpected load response: {}".format(user))
            except Exception as e:
                Logger.log("CDN GETPROFILEINFO ERROR: Failed to load user '{}' for superuser request: {}".format(userID, e))
                return JSONRes.ambiguousError()
    
    if request.args.get('includeLogs', 'false').lower() == 'true':
        try:
            user.getAuditLogs()
        except Exception as e:
            Logger.log("USERPROFILE INFO ERROR: Failed to retrieve audit logs for user '{}' (will skip); error: {}".format(user.username, e))
            user.logs = None
    
    info = {
        'username': user.username,
        'email': user.email,
        'fname': user.fname,
        'lname': user.lname,
        'role': user.role,
        'contact': user.contact,
        'pfp': True if user.pfp else False,
        'lastLogin': user.lastLogin,
        'created': user.created
    }
    
    if isinstance(user.logs, list):
        info['logs'] = [log.represent() for log in user.logs]
    else:
        info['logs'] = None
    
    return JSONRes.new(
        code=200,
        msg="Information retrieved successfully.",
        info=info
    )

@cdnBP.route('/pfp/<userID>')
@checkSession(strict=True, provideUser=True)
def getProfilePicture(user: User, userID: str=None):
    # Carry out authorisation checks
    if userID != user.id:
        if not user.superuser:
            # User is not the same as the requested user and is not a superuser
            return JSONRes.unauthorised()
        else:
            # User is a superuser, so we can load the requested user
            try:
                user = User.load(id=userID)
                if user is None:
                    return JSONRes.new(404, "User not found.")
                if not isinstance(user, User):
                    raise Exception("Unexpected load response: {}".format(user))
            except Exception as e:
                Logger.log("CDN GETPFP ERROR: Failed to load user '{}' for superuser request: {}".format(userID, e))
                return JSONRes.ambiguousError()
    
    if not user.pfp:
        return JSONRes.new(404, "No profile picture found for this user.")
    
    file = File(user.pfp, "FileStore")
    res = None
    if request.args.get('raw', 'false').lower() == 'true':
        res = FileManager.prepFile(file=file)
        if isinstance(res, str):
            if res == "ERROR: File does not exist.":
                user.pfp = None
                user.save()
                return JSONRes.new(404, "No profile picture found for this user.")
            else:
                Logger.log("CDN GETPFP ERROR: Failed to prepare file for user '{}': {}".format(user.id, res))
                return JSONRes.ambiguousError()
        
        res = make_response(send_file(file.path()))
    else:
        try:
            url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
            if url.startswith("ERROR"):
                if url == "ERROR: File does not exist.":
                    user.pfp = None
                    user.save()
                    return JSONRes.new(404, "No profile picture found for this user.")
                else:
                    raise Exception(url)
            
            res = make_response(redirect(url))
        except Exception as e:
            Logger.log("CDN GETPFP ERROR: Failed to generate signed URL for user '{}': {}".format(user.id, e))
            return JSONRes.ambiguousError()
    
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
    res.headers["Pragma"] = "no-cache"
    res.headers["Expires"] = "0"
    return res

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
    
    if os.environ.get("DEBUG_MODE", "False") == "True":
        # In debug mode, serve the file directly from the filesystem
        res = FileManager.prepFile(file=file)
        if isinstance(res, str):
            if res == "ERROR: File does not exist.":
                return JSONRes.new(404, "Requested file not found.")
            else:
                Logger.log("CDN GETARTEFACT ERROR: Failed to prepare file for delivery; response: {}".format(res))
                return JSONRes.ambiguousError()
        
        try:
            return send_file(file.path())
        except Exception as e:
            Logger.log("CDN GETARTEFACT ERROR: Failed to send file '{}' for artefact '{}'; error: {}".format(filename, artefactId, e))
            return JSONRes.ambiguousError()
    
    # Attempt to generate signed URL
    try:
        url = file.getSignedURL(expiration=datetime.timedelta(minutes=10)) # note: useCache=True
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

@cdnBP.route('/people/<figureID>')
@checkSession(strict=True)
def getFaceImage(figureID):
    """
    Serve a face image from the 'people' store.
    """
    file = File('{}.jpg'.format(figureID), "people")
    
    if os.environ.get("DEBUG_MODE", "False") == "True":
        # In debug mode, serve the file directly from the filesystem
        res = FileManager.prepFile(file=file)
        if isinstance(res, str):
            if res == "ERROR: File does not exist.":
                return JSONRes.new(404, "Requested file not found.")
            else:
                Logger.log("CDN GETFACEIMAGE ERROR: Failed to prepare file for delivery; response: {}".format(res))
                return JSONRes.ambiguousError()
        
        try:
            return send_file(file.path())
        except Exception as e:
            Logger.log("CDN GETFACEIMAGE ERROR: Failed to send headshot for figure '{}'; error: {}".format(figureID, e))
            return JSONRes.ambiguousError()
    
    try:
        url = file.getSignedURL(expiration=datetime.timedelta(seconds=60))
        
        if url.startswith("ERROR"):
            if url == "ERROR: File does not exist.":
                return JSONRes.new(404, "Requested file not found.")
            else:
                raise Exception(url)
        
        res = make_response(redirect(url))

        # Set headers to force revalidation and prevent caching
        res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        res.headers["Pragma"] = "no-cache"
        res.headers["Expires"] = "0"
        
        return res
    except Exception as e:
        Logger.log("CDN GETFACEIMAGE ERROR: Unexpected error in redirecting to a signed URL for figure '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()

@cdnBP.route('/asset/<filename>')
@checkSession(strict=True)
def getAsset(filename):
    """
    Serve a public file from the 'FileStore' store (via /FileStore).
    """    
    file = File(filename, "FileStore")
    
    if os.environ.get("DEBUG_MODE", "False") == "True":
        # In debug mode, serve the file directly from the filesystem
        res = FileManager.prepFile(file=file)
        if isinstance(res, str):
            if res == "ERROR: File does not exist.":
                return JSONRes.new(404, "Requested file not found.")
            else:
                Logger.log("CDN GETASSET ERROR: Failed to prepare file for delivery; response: {}".format(res))
                return JSONRes.ambiguousError()
        
        try:
            return send_file(file.path())
        except Exception as e:
            Logger.log("CDN GETASSET ERROR: Failed to send asset '{}'; error: {}".format(filename, e))
            return JSONRes.ambiguousError()
    
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
        Logger.log("CDN GETASSET ERROR: Unexpected error in redirecting to a signed URL for asset '{}'; error: {}".format(filename, e))
        return JSONRes.ambiguousError()

@cdnBP.route('/catalogue')
@checkSession(strict=True)
@cache
def getAllCategoriesWithArtefacts():
    """
    Retrieves all human-figure artefacts grouped by category and all books with their associated 
    meeting-minute artefacts.

    Key Operations:
    1. Loads all artefacts (without metadata) into a lookup map keyed by artefact ID.
       - Differentiates artefacts by `artType` ("hf" for human-figure, "mm" for meeting-minute).
    2. Loads all categories and attaches their human-figure artefacts (`artType = "hf"`).
    3. Loads all books and attaches their meeting-minute artefacts (`artType = "mm"`) via `mmIDs`.
    4. Handles exceptions at every loading or lookup stage.
    5. Requires session authentication.

    Response Format:
    ```json
    {
        "categories": {
            "CategoryID1": {
                "name": "Category Name",
                "description": "Optional description",
                "members": [
                    { "id": "...", "name": "...", "image": "...", "description": "..." },
                    ...
                ]
            },
            ...
        },
        "books": [
            {
                "id": "...",
                "title": "...",
                "subtitle": "...",
                "mmArtefacts": [
                    { "id": "...", "name": "...", "description": "..." },
                    ...
                ]
            },
            ...
        ]
    }
    ```
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
        result[cat.id] = {
            "name": cat.name,
            "description": cat.description or "Unavailable",
            "members": artefactsList
        }

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
    
@cdnBP.route('/collectionMemberIDs/<colID>')
@checkSession(strict=True)
@cache
def getCollectionMemberIDs(colID):
    """
    Returns the list of all artefact IDs in a collection, which can be a Book, Category, or Batch.

    This endpoint:
    - Attempts to load the collection sequentially as a Book, Category, or Batch by `colID`.
    - For Books: returns the `mmIDs` (meeting-minute artefact IDs).
    - For Categories: returns the keys of `members` (human-figure artefact IDs).
    - For Batches: returns the keys of `artefacts` (all artefact IDs in the batch).
    - Logs any errors and returns a generic error response on failure.

    Response Format:
    - Success (200):
    ```json
    ["artefactID1", "artefactID2", ...]
    ````

    * Not found (404):

    ```json
    "Collection not found."
    ```

    * Error (ambiguous):

    ```json
    "An error occurred."
    ```

    """

    try:
        # Try Book
        book = Book.load(id=colID)
        if isinstance(book, Book):
            return JSONRes.new(200, "Retrieval success.", data=book.mmIDs)
        
        # Try Category
        cat = Category.load(id=colID)
        if isinstance(cat, Category):
            return JSONRes.new(200, "Retrieval success.", data=list(cat.members.keys()))
        
        # Try Batch
        batch = Batch.load(id=colID)
        if isinstance(batch, Batch):
            return JSONRes.new(200, "Retrieval success.", data=list(batch.artefacts.keys()))
        
        # None found
        return JSONRes.new(404, "Collection not found.")
    
    except Exception as e:
        Logger.log("CDN GETCOLLECTIONMEMBERIDS ERROR: Failed to load collection members - {}".format(e))
        return JSONRes.ambiguousError()
    
@cdnBP.route('/collection/<colID>')
@checkSession(strict=True)
@cache
def getCollectionDetails(colID):
    """
    Returns detailed information about a collection (book, category, or batch) including all its member artefacts.
    
    This endpoint:
    - Attempts to load the collection as each type (book, category, batch)
    - Loads all artefacts for resolving member references
    - Returns the collection details with full artefact information
    
    Response Format (varies by collection type):
    - For Books:
    ```json
    {
        "type": "book",
        "id": "...",
        "title": "...",
        "subtitle": "...",
        "mmArtefacts": [
            { "id": ..., "name": ..., "description": ..., "image": ..., ... },
            ...
        ]
    }
    ```
    - For Categories:
    ```json
    {
        "type": "category",
        "id": "...",
        "name": "...",
        "description": "...",
        "members": [
            { "id": ..., "name": ..., "image": ..., "description": ..., ... },
            ...
        ]
    }
    ```
    - For Batches:
    ```json
    {
        "type": "batch",
        "id": "...",
        ... (other batch fields),
        "artefacts": [
            { "id": ..., "name": ..., ... (other artefact fields) },
            ...
        ]
    }
    ```
    """
    
    # First load all artefacts for reference
    try:
        all_artefacts = Artefact.load(includeMetadata=False) or []
        artefactMap: Dict[str, Artefact] = {art.id: art for art in all_artefacts}
    except Exception as e:
        Logger.log("CDN GETCOLLECTIONDETAILS ERROR: Failed to load artefacts - {}".format(e))
        return JSONRes.ambiguousError()
    
    # Try loading as each collection type
    try:
        # Try Book
        book = Book.load(id=colID)
        if isinstance(book, Book):
            mm_details = []
            for mmID in book.mmIDs:
                art = artefactMap.get(mmID)
                if art and art.artType == "mm":
                    mm_details.append({
                        "id": art.id,
                        "name": art.name,
                        "description": art.description or "Unavailable",
                        "image": art.image
                    })
            
            return JSONRes.new(200, "Retrieval success.", data={
                "type": "book",
                "id": book.id,
                "title": book.title,
                "subtitle": book.subtitle or "Unavailable",
                "mmArtefacts": mm_details
            })
        
        # Try Category
        cat = Category.load(id=colID)
        if isinstance(cat, Category):
            members = []
            for artefact_id, _ in cat.members.items():
                art = artefactMap.get(artefact_id)
                if art and art.artType == "hf":
                    members.append({
                        "id": art.id,
                        "name": art.name,
                        "image": art.image,
                        "description": art.description or "Unavailable"
                    })
            
            return JSONRes.new(200, "Retrieval success.", data={
                "type": "category",
                "id": cat.id,
                "name": cat.name,
                "description": cat.description or "Unavailable",
                "members": members
            })
        
        # Try Batch
        batch = Batch.load(id=colID)
        if isinstance(batch, Batch):
            artefacts = []
            for art_id in batch.artefacts.keys():
                art = artefactMap.get(art_id)
                if art:
                    artefact_data = {
                        "id": art.id,
                        "name": art.name,
                        "artType": art.artType,
                        "image": art.image,
                        "description": art.description or "Unavailable"
                    }
                    artefacts.append(artefact_data)
            
            return JSONRes.new(200, "Retrieval success.", data={
                "type": "batch",
                "id": batch.id,
                "name": batch.name or "Name Unavailable",
                "artefacts": artefacts
            })
        
        return JSONRes.new(404, "Collection not found.")
    
    except Exception as e:
        Logger.log("CDN GETCOLLECTIONDETAILS ERROR: Failed to load collection - {}".format(e))
        return JSONRes.ambiguousError()
    
@cdnBP.route('/retrieveAssociationInfo/<artID>')
@checkSession(strict=True)
@cache
def getAssociationInfo(artID):
    try:
        art = Artefact.load(id=artID)
        if not isinstance(art, Artefact):
            return JSONRes.new(404, "Artefact not found.")

        mm = art.metadata.isMM()
        results = []

        if mm:
            allBooks = Book.load()
            for book in allBooks:
                results.append({
                    "type": "book",
                    "id": book.id,
                    "description": book.subtitle,
                    "name": book.title or "Unavailable",
                    "isMember": artID in book.mmIDs
                })
        else:
            allCat = Category.load()
            for cat in allCat:
                results.append({
                    "type": "category",
                    "id": cat.id,
                    "description": cat.description,
                    "name": cat.name or "Unavailable",
                    "isMember": artID in cat.members
                })

        return JSONRes.new(200, "Retrieval Successful", data=results)

    except Exception as e:
        Logger.log("CDN GETASSOCIATEDINFO ERROR: Failed to fetch associations by artefact ID - {}".format(e))
        return JSONRes.ambiguousError()


@cdnBP.route('/artefactMetadata/<artID>')
@checkSession(strict=True)
@cache(lsInvalidator='artefactMetadata')
def getArtefactMetedata(artID):
    """
    Returns the metadata of an artefact.
    """
    try:
        art = Artefact.load(id=artID, includeMetadata=True)
        if not isinstance(art, Artefact):
            return JSONRes.new(404, "Artefact not found.")
        
        return JSONRes.new(200, "Retrieval success.", data=art.represent())

    except Exception as e:
        Logger.log("CDN GETARTEFACTMETADATA ERROR: Failed to load artefact metadata - {}".format(e))
        return JSONRes.ambiguousError()
