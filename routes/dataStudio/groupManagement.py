from flask import Blueprint, request
from utils import JSONRes
from services import Logger, LiteStore
from decorators import jsonOnly, enforceSchema, checkAPIKey
from sessionManagement import checkSession
from schemas import User, Category, Book

grpBP = Blueprint('group', __name__, url_prefix="/group")

@grpBP.route('/create', methods=['POST'])
@checkAPIKey
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
            user.newLog("Category Created", "Created category '{}'".format(title))
            
        LiteStore.set("catalogue", True)
        return JSONRes.new(200, "{} created successfully".format(groupType.capitalize()))
        
    except Exception as e:
        Logger.log("GROUPMANAGEMENT CREATEGROUP ERROR: Failed to create {} with title'{}': {}".format(groupType, title, e))
        return JSONRes.ambiguousError()
    
@grpBP.route('/updateDetails', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("collectionID", str),
    ("name", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 50),
    ("description", lambda x: isinstance(x, str) and len(x.strip()) <= 200)
)
@checkSession(strict=True, provideUser=True)
def updateDetails(user: User):
    collectionID = request.json.get("collectionID")
    newName = request.json.get("name")
    newDescription = request.json.get("description")

    try:
        collection = Book.load(id=collectionID)
        if isinstance(collection, Book):
            changes = []

            if newName and collection.title != newName:
                collection.title = newName.strip()
                changes.append("Name")

            if newDescription is not None and collection.subtitle != newDescription:
                collection.subtitle = newDescription.strip() if newDescription else None
                changes.append("Description")

            if changes:
                collection.save()
                logName = newName if newName else collection.title
                user.newLog(f"Group {logName} Update", ", ".join(changes) + " updated.")

                LiteStore.set("catalogue", True)
                LiteStore.set("collectionMemberIDs", True)
                LiteStore.set("collectionDetails", True)
                LiteStore.set("associationInfo", True)

                return JSONRes.new(200, "Updated successfully")

            return JSONRes.new(200, "No changes made to the collection")

        collection = Category.load(id=collectionID)
        if isinstance(collection, Category):
            changes = []

            if newName and collection.name != newName:
                collection.name = newName.strip()
                changes.append("Name")

            if newDescription is not None and collection.description != newDescription:
                collection.description = newDescription.strip() if newDescription else None
                changes.append("Description")

            if changes:
                collection.save()
                logName = newName if newName else collection.name
                user.newLog(f"Group {logName} Update", ", ".join(changes) + " updated.")

                LiteStore.set("catalogue", True)
                LiteStore.set("collectionMemberIDs", True)
                LiteStore.set("collectionDetails", True)
                LiteStore.set("associationInfo", True)

                return JSONRes.new(200, "Updated successfully")

            return JSONRes.new(200, "No changes made to the collection")

        return JSONRes.new(404, "Collection not found")

    except Exception as e:
        Logger.log(f"GROUPMANAGEMENT UPDATE ERROR: Failed to update collection {collectionID}: {e}")
        return JSONRes.ambiguousError()

@grpBP.route('/delete', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("groupID", str),
)
@checkSession(strict=True, provideUser=True)
def deleteGroup(user: User):
    groupID = request.json['groupID']
    
    try:
        group = Book.load(id=groupID)
        
        if not isinstance(group, Book):
            group = Category.load(id=groupID)
            
            if not isinstance(group, Category):
                return JSONRes.new(404, "Group not found")
            
    except Exception as e:
        Logger.log("GROUPMANAGEMENT DELETEGROUP ERROR: Failed to load group {}: {}".format(groupID, e))
        return JSONRes.ambiguousError()
    
    try:
        group.destroy()
    except Exception as e:
        Logger.log("GROUPMANAGEMENT DELETEGROUP ERROR: Failed to delete group {}: {}".format(groupID, e))
        return JSONRes.ambiguousError
    
    LiteStore.set("catalogue", True)
    LiteStore.set("associationInfo", True)
            
    return JSONRes.new(200, "Group deleted successfully")
