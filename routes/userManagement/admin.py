from typing import List, Dict, Tuple
from flask import Blueprint, request, redirect, url_for
from utils import JSONRes, ResType, Extensions
from services import Universal, Logger
from decorators import jsonOnly, checkAPIKey, enforceSchema
from schemas import User
from sessionManagement import checkSession, requireSuperuser

adminBP = Blueprint("admin", __name__, url_prefix="/admin")

@adminBP.route("/listUsers", methods=['GET'])
@checkAPIKey
@checkSession(strict=True, provideUser=True)
@requireSuperuser
def listUsers(user: User):
    try:
        users: List[User] = User.load()
        
        if not isinstance(users, list):
            raise Exception("Unexpected load response in retrieving users; response: {}".format(users))
        
        # Remove superuser from the list
        users = list(filter(lambda u: u.id != user.id, users))
        
        data = Extensions.sanitiseData(
            {u.id: u.represent() for u in users},
            disallowedKeys=['authToken', 'pwd', 'originRef', 'resetKey', 'pfp', 'superuser']
        )
        
        return JSONRes.new(200, "Users retrieved successfully", ResType.SUCCESS, users=data)
    except Exception as e:
        Logger.log("ADMIN LISTUSERS ERROR: Failed to retrieve users data; error: {}".format(e))
        return JSONRes.ambiguousError()