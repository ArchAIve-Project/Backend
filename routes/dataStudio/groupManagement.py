from flask import Blueprint, request
from utils import JSONRes
from services import Logger, LiteStore
from decorators import jsonOnly, enforceSchema, checkAPIKey
from schemas import User, Category, Book
from sessionManagement import checkSession

grpBP = Blueprint('group', __name__, url_prefix="/studio/group")

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
        mmIds = []
        groupID = None

        if groupType == "book":
            book = Book(
                title=title,
                subtitle=subtitle,
                mmIDs=mmIds
            )
            book.save()
            user.newLog("Book Created", "Created book with title '{}'.".format(title))
            groupID = book.id
        elif groupType == "category":
            category = Category(
                name=title,
                description=subtitle
            )
            category.save()
            user.newLog("Category Created", "Created category with name '{}'.".format(title))
            groupID = category.id
        
        LiteStore.set("catalogue", True)
        return JSONRes.new(200, "Group created successfully.", groupID=groupID)
    except Exception as e:
        Logger.log("GROUPMANAGEMENT CREATEGROUP ERROR: Failed to create {} with title'{}': {}".format(groupType, title, e))
        return JSONRes.ambiguousError()
    
@grpBP.route('/updateDetails', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("collectionID", str),
    ("name", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 50, None),
    ("description", lambda x: isinstance(x, str) and len(x.strip()) <= 200, None)
)
@checkSession(strict=True, provideUser=True)
def updateDetails(user: User):
    collectionID = request.json.get("collectionID")
    newName = request.json.get("name")
    newDescription = request.json.get("description")
    
    try:
        collection: Book | Category | None = Book.load(id=collectionID) or Category.load(id=collectionID)
        if not collection:
            return JSONRes.new(404, "No such group with that ID.")
        
        updatesMade = False
        if isinstance(collection, Book):
            changes = []
            if newName and collection.title != newName:
                collection.title = newName.strip()
                changes.append("Title")
            if newDescription and collection.subtitle != newDescription:
                collection.subtitle = newDescription.strip() if newDescription else None
                changes.append("Description")
            
            if changes:
                collection.save()
                user.newLog("Book Updated", "{} details changed.".format(", ".join(changes)))
                updatesMade = True
        elif isinstance(collection, Category):
            changes = []
            if newName and collection.name != newName:
                collection.name = newName.strip()
                changes.append("Name")
            if newDescription and collection.description != newDescription:
                collection.description = newDescription.strip() if newDescription else None
                changes.append("Description")

            if changes:
                collection.save()
                user.newLog("Category Updated", "{} details changed.".format(", ".join(changes)))
                updatesMade = True
        
        if updatesMade:
            LiteStore.set("catalogue", True)
            LiteStore.set("collectionMemberIDs", True)
            LiteStore.set("collectionDetails", True)
            LiteStore.set("associationInfo", True)
            
            return JSONRes.new(200, "Group details updated successfully.")
        else:
            return JSONRes.new(200, "No changes made to the group details.")
    except Exception as e:
        Logger.log("GROUPMANAGEMENT UPDATE ERROR: Failed to update group {}: {}".format(collectionID, e))
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
        group: Book | Category | None = Book.load(id=groupID) or Category.load(id=groupID)
        if not group:
            return JSONRes.new(404, "No such group with that ID.")
    except Exception as e:
        Logger.log("GROUPMANAGEMENT DELETEGROUP ERROR: Failed to load group {}: {}".format(groupID, e))
        return JSONRes.ambiguousError()
    
    try:
        group.destroy()
        user.newLog("Group Deleted", "Deleted group with name/title '{}'".format(group.name if isinstance(group, Category) else group.title))
    except Exception as e:
        Logger.log("GROUPMANAGEMENT DELETEGROUP ERROR: Failed to delete group {}: {}".format(groupID, e))
        return JSONRes.ambiguousError()
    
    LiteStore.set("catalogue", True)
    LiteStore.set("associationInfo", True)
    
    return JSONRes.new(200, "Group deleted successfully.")