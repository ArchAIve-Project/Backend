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
            user.newLog("BookCreated", "Created book '{0}'".format(title))
            
        elif groupType == "category":
            Category(
                name=title,
                description=subtitle
            ).save()
            user.newLog("CategoryCreated", "Created category '{0}'".format(title))
            
        LiteStore.set("catalogue", True)
        return JSONRes.new(200, "{} created successfully".format(groupType.capitalize()))
        
    except Exception as e:
        Logger.log("GROUPMANAGEMENT CREATEGROUP ERROR: Failed to create {} '{}': {}".format(
            groupType if "groupType" in locals() else "unknown",
            title if "title" in locals() else "unknown",
            str(e)
        ))
        return JSONRes.ambiguousError()