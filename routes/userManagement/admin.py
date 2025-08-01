import re
from typing import List, Dict, Tuple
from flask import Blueprint, request, redirect, url_for
from utils import JSONRes, ResType, Extensions
from services import Universal, Logger
from decorators import jsonOnly, checkAPIKey, enforceSchema, Param
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

@adminBP.route("/createUser", methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "username",
        lambda x: isinstance(x, str) and len(x) > 3 and x.isalpha(),
        invalidRes=JSONRes.new(400, "Username must be at least 3 characters, without any spaces or special characters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "email",
        lambda x: isinstance(x, str) and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x.strip()),
        invalidRes=JSONRes.new(400, "Invalid email.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "role",
        lambda x: isinstance(x, str) and 2 <= len(x) <= 20,
        invalidRes=JSONRes.new(400, "Role must be between 2 and 20 characters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "fname",
        lambda x: isinstance(x, str) and len(x) > 2 and x.replace(' ', '').isalpha(),
        invalidRes=JSONRes.new(400, "First name must be at least 2 characters and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "lname",
        lambda x: isinstance(x, str) and len(x) > 1 and x.replace(' ', '').isalpha(),
        invalidRes=JSONRes.new(400, "Last name must be at least 1 character and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "contact",
        lambda x: isinstance(x, str) and len(x.replace(' ', '').strip()) == 8 and x.replace(' ', '').strip().isdigit(),
        invalidRes=JSONRes.new(400, "Contact number must be 8 digits.", ResType.USERERROR, serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
@requireSuperuser
def createUser(user: User):
    username: str = request.json.get('username')
    email: str = request.json.get('email')
    role: str = request.json.get('role')
    fname: str = request.json.get('fname')
    lname: str = request.json.get('lname')
    contact: str = request.json.get('contact').replace(' ', '').strip()
    
    # Create the new user
    u = User()