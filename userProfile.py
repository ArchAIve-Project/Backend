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
def getProfileInfo(user: User):
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