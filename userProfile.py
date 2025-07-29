import re
from flask import Blueprint, url_for, request
from utils import JSONRes, ResType
from services import Universal, Logger, Encryption
from decorators import jsonOnly, enforceSchema, checkAPIKey
from sessionManagement import checkSession
from models import User, AuditLog

profileBP = Blueprint('profile', __name__, url_prefix='/profile')

@profileBP.route('/info', methods=['GET'])
@checkAPIKey
@checkSession(strict=True, provideUser=True)
def getInfo(user: User):
    return JSONRes.new(
        code=200,
        msg="Information retrieved successfully.",
        info={
            'username': user.username,
            'email': user.email,
            'fname': user.fname,
            'lname': user.lname,
            'role': user.role,
            'contact': user.contact,
            'lastLogin': user.lastLogin,
            'created': user.created
        }
    )

@profileBP.route('/update', methods=['POST'])
@checkAPIKey
@enforceSchema(
    ("username", lambda x: isinstance(x, str) and len(x) > 0, None),
    ("fname", lambda x: isinstance(x, str) and len(x) > 0, None),
    ("lname", lambda x: isinstance(x, str) and len(x) > 0, None),
    ("contact", lambda x: isinstance(x, str) and x.isdigit(), None),
    ("email", lambda x: isinstance(x, str) and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x), None)
)
@checkSession(strict=True, provideUser=True)
def update(user: User):
    username = request.json.get('username', user.username).strip()
    fname = request.json.get('fname', user.fname).strip()
    lname = request.json.get('lname', user.lname).strip()
    contact = request.json.get('contact', user.contact).strip()
    email = request.json.get('email', user.email).strip()
    
    # Check username uniqueness
    if username != user.username:
        conflictingUser = None
        try:
            conflictingUser = User.load(username=username)
            print(conflictingUser)
            if isinstance(conflictingUser, User):
                return JSONRes.new(400, "Username already exists.", ResType.USERERROR)
        except Exception as e:
            Logger.log("USERPROFILE UPDATE ERROR: Failed to load user with username '{}' for uniqueness check; error: {}".format(username, e))
            return JSONRes.ambiguousError()
    
    # Check email uniqueness
    if email != user.email:
        conflictingUser = None
        try:
            conflictingUser = User.load(email=email)
            if isinstance(conflictingUser, User):
                return JSONRes.new(400, "Email already exists.", ResType.USERERROR)
        except Exception as e:
            Logger.log("USERPROFILE UPDATE ERROR: Failed to load user with email '{}' for uniqueness check; error: {}".format(email, e))
            return JSONRes.ambiguousError()
    
    # Update user details
    changes = []
    if username != user.username:
        user.username = username
        changes.append("Username")
    if fname != user.fname:
        user.fname = fname
        changes.append("First Name")
    if lname != user.lname:
        user.lname = lname
        changes.append("Last Name")
    if contact != user.contact:
        user.contact = contact
        changes.append("Contact")
    if email != user.email:
        user.email = email
        changes.append("Email")
    
    if changes:
        user.save()
        user.newLog("Profile Update", "{} details updated.".format(", ".join(changes)))
    else:
        return JSONRes.new(200, "No changes made to the profile.")
    
    return JSONRes.new(200, "Profile updated successfully.")