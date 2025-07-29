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
        username=user.username,
        email=user.email,
        fname=user.fname,
        lname=user.lname,
        role=user.role,
        contact=user.contact,
        lastLogin=user.lastLogin,
        created=user.created
    )

@profileBP.route('/update', methods=['POST'])
@checkAPIKey
@checkSession(strict=True, provideUser=True)
def update(user: User):
    username = request.json.get('username', user.username)
    if username != user.username:
        existingUser = None
        try:
            existingUser = User.load(username=username)
        except Exception as e:
            Logger.log("USERPROFILE UPDATE ERROR: Failed to load user for username '{}' uniqueness check; error: {}".format(username, e))
    
    fname = request.json.get('fname', user.fname)
    lname = request.json.get('lname', user.lname)
    contact = request.json.get('contact', user.contact)
    email = request.json.get('email', user.email)
    return "lmao"