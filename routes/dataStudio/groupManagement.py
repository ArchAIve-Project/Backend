from flask import Blueprint, request
from utils import JSONRes
from services import Logger, LiteStore
from decorators import jsonOnly, enforceSchema, checkAPIKey
from sessionManagement import checkSession
from schemas import User, Category, Book

grpBP = Blueprint('group', __name__, url_prefix="/group")

@grpBP.route('/create', methods=['POST'])
# @checkAPIKey
@jsonOnly
@enforceSchema(
    ("title", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 25),
    ("subtitle", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 100),
    ("type", lambda x: isinstance(x, str) and x in ["book", "category"])
)
@checkSession(strict=True, provideUser=True)
def createGroup(user: User):
    try:
        groupType = request.json["type"]
        title = request.json["title"].strip()
        subtitle = request.json.get("subtitle", "").strip() or None
        mmIds=[]

        if groupType == "book":
            Book(
                title=title,
                subtitle=subtitle,
                mmIDs=mmIds
            ).save()
            user.newLog("BookCreated", "Created book '{}'".format(title))
            
        elif groupType == "category":
            Category(
                name=title,
                description=subtitle
            ).save()
            user.newLog("CategoryCreated", "Created category '{}'".format(title))
            
        LiteStore.set("catalogue", True)
        return JSONRes.new(200, "{} created successfully".format(groupType.capitalize()))
        
    except Exception as e:
        Logger.log("GROUPMANAGEMENT CREATEGROUP ERROR: Failed to create {} '{}': {}".format(groupType, title, e))
        return JSONRes.ambiguousError()
    
@grpBP.route('/updateDetails', methods=['POST'])
@jsonOnly
@enforceSchema(
    ("collectionID", lambda x: isinstance(x, str) and len(x.strip()) > 0),
    ("name", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 50),
    ("description", lambda x: isinstance(x, str) and len(x.strip()) <= 200)
)
@checkSession(strict=True, provideUser=True)
def updateDetails(user):
    collectionID = request.json.get("collectionID")
    newName = request.json.get("name")
    newDescription = request.json.get("description")
    changes = []
    
    try:
        collection = Book.load(id=collectionID)
        isBook = True
        
        if not isinstance(collection, Book):
            collection = Category.load(id=collectionID)
            isBook = False
            
            if not isinstance(collection, Category):
                return JSONRes.new(404, "Collection not found")
        
        currentName = collection.title if isBook else collection.name
        if newName and currentName != newName:
            if isBook:
                collection.title = newName.strip()
            else:
                collection.name = newName.strip()
            changes.append("Name")
            
        currentDesc = collection.subtitle if isBook else collection.description
        if newDescription is not None and currentDesc != newDescription:
            if isBook:
                collection.subtitle = newDescription.strip() if newDescription else None
            else:
                collection.description = newDescription.strip() if newDescription else None
            changes.append("Description")
        
        if changes:
            collection.save()
            user.newLog("Group {} Update".format(newName), ", ".join(changes) + " updated.")

            LiteStore.set("catalogue", True)
            LiteStore.set("collectionMemberIDs", True)
            LiteStore.set("collectionDetails", True)
            LiteStore.set("associationInfo", True)
            
            return JSONRes.new(200, "Updated successfully")
            
        return JSONRes.new(200, "No changes made to the collection")
        
    except Exception as e:
        Logger.log("GROUPMANAGEMENT UPDATE ERROR: Failed to update collection {}: {}".format(collectionID, e))
        return JSONRes.ambiguousError()
