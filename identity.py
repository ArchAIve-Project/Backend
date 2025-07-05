from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from services import Logger, Encryption, Universal
from decorators import jsonOnly, enforceSchema
from models import User

identityBP = Blueprint('identity', __name__, url_prefix="/auth")

@identityBP.route('/login', methods=['POST'])
@jsonOnly
@enforceSchema(
    ("usernameOrEmail", str),
    ("password", str)
)
def login():
    usernameOrEmail: str = request.json['usernameOrEmail'].strip()
    password: str = request.json['password'].strip()
    
    # Retrieve user by username or email
    try:
        user = User.load(username=usernameOrEmail, email=usernameOrEmail)
        if not isinstance(user, User):
            return JSONRes.new(401, "Invalid credentials.", ResType.USERERROR)
    except Exception as e:
        Logger.log("IDENTITY LOGIN ERROR: Failed to retrieve user; error: {}".format(e))
        return JSONRes.ambiguousError()
    
    if not Encryption.verifySHA256(password, user.pwd):
        return JSONRes.new(401, "Invalid credentials.", ResType.USERERROR)
    
    # Generate auth token
    authToken = Universal.generateUniqueID(customLength=12)
    user.authToken = authToken
    user.lastLogin = Universal.utcNowString()
    user.save()
    
    # Update user session
    session['accID'] = user.id
    session['username'] = user.username
    session['authToken'] = user.authToken
    session['sessionStart'] = user.lastLogin
    session['superuser'] = user.superuser
    
    return JSONRes.new(200, "Login successful.")