import re
from typing import List, Dict, Tuple
from flask import Blueprint, request, redirect, url_for
from utils import JSONRes, ResType, Extensions
from services import Universal, Logger, Encryption, ThreadManager
from fm import File, FileManager
from decorators import jsonOnly, checkAPIKey, enforceSchema, Param
from schemas import User, AuditLog
from sessionManagement import checkSession
from emailCentre import EmailCentre, AdminPasswordResetAlert, AccountConfirmationAlert

adminBP = Blueprint("admin", __name__, url_prefix="/admin")

@adminBP.route("/listUsers", methods=['GET'])
@checkAPIKey
@checkSession(strict=True, provideUser=True, requireSuperuser=True)
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
        
        return JSONRes.new(200, "Users retrieved successfully.", ResType.SUCCESS, users=data)
    except Exception as e:
        Logger.log("ADMIN LISTUSERS ERROR: Failed to retrieve users data; error: {}".format(e))
        return JSONRes.ambiguousError()

@adminBP.route("/createUser", methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "username",
        lambda x: isinstance(x, str) and len(x) >= 3 and x.isalnum() and len(x) <= 12,
        invalidRes=JSONRes.new(400, "Username must be between 3 and 12 characters, without any spaces or special characters.", ResType.USERERROR, serialise=False)
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
        lambda x: isinstance(x, str) and len(x) >= 2 and x.replace(' ', '').isalpha() and len(x) <= 15,
        invalidRes=JSONRes.new(400, "First name must be between 2 and 15 characters and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "lname",
        lambda x: isinstance(x, str) and len(x) >= 1 and x.replace(' ', '').isalpha() and len(x) <= 15,
        invalidRes=JSONRes.new(400, "Last name must be between 1 and 15 characters and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "contact",
        lambda x: isinstance(x, str) and len(x.replace(' ', '').strip()) == 8 and x.replace(' ', '').strip().isdigit(),
        invalidRes=JSONRes.new(400, "Contact number must be 8 digits.", ResType.USERERROR, serialise=False)
    )
)
@checkSession(strict=True, requireSuperuser=True)
def createUser():
    username: str = request.json.get('username').strip()
    email: str = request.json.get('email').strip()
    role: str = request.json.get('role').strip()
    fname: str = request.json.get('fname').strip()
    lname: str = request.json.get('lname').strip()
    contact: str = request.json.get('contact').replace(' ', '').strip()
    
    # Create the new user
    userPass = Universal.generateUniqueID(customLength=6)
    u = User(
        username=username,
        email=email,
        pwd=Encryption.encodeToSHA256(userPass),
        fname=fname,
        lname=lname,
        role=role,
        contact=contact
    )
    
    try:
        u.save()
    except Exception as e:
        # Possible key violation
        e = str(e)
        if e.startswith("USER SAVE ERROR: Integrity violation: "):
            violation = e[len("USER SAVE ERROR: Integrity violation: "):]
            if violation == "Username":
                return JSONRes.new(400, "Username already exists.", ResType.USERERROR)
            elif violation == "Email":
                return JSONRes.new(400, "Email already exists.", ResType.USERERROR)
        
        Logger.log("ADMIN CREATEUSER ERROR: Failed to create new user with username '{}'; error: {}".format(username, e))
        return JSONRes.ambiguousError()
    
    ThreadManager.defaultProcessor.addJob(EmailCentre.dispatch, AccountConfirmationAlert(u, userPass))
    
    Logger.log("ADMIN CREATEUSER: New user created with username '{}' and ID '{}'.".format(username, u.id))
    
    return JSONRes.new(200, "User created successfully. Login instructions have been dispatched to the user.", ResType.SUCCESS, newUserID=u.id)

@adminBP.route("/deleteUser", methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "userID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Target user ID required.", serialise=False)
    )
)
@checkSession(strict=True, provideUser=True, requireSuperuser=True)
def deleteUser(user: User):
    userID: str = request.json.get("userID").strip()
    if userID == user.id:
        return JSONRes.new(400, "Cannot delete your own account.", ResType.USERERROR)
    
    try:
        targetUser: User = User.load(userID, withLogs=True)
        if not isinstance(targetUser, User):
            return JSONRes.new(404, "User not found.", ResType.USERERROR)
    except Exception as e:
        Logger.log("ADMIN DELETEUSER ERROR: Failed to load user with ID '{}'; error: {}".format(userID, e))
        return JSONRes.ambiguousError()
    
    try:
        if targetUser.pfp:
            file = File(targetUser.pfp, 'FileStore')
            res = FileManager.delete(file=file)
            if isinstance(res, str):
                Logger.log("ADMIN DELETEUSER ERROR: Failed to delete profile picture for user '{}' (ID: {}); error: {}".format(targetUser.username, targetUser.id, res))
        
        if isinstance(targetUser.logs, list):
            for log in targetUser.logs:
                if isinstance(log, AuditLog):
                    log.destroy()
        
        targetUser.destroy()
        Logger.log("ADMIN DELETEUSER: User with ID '{}' and username '{}' deleted successfully.".format(userID, targetUser.username))
    except Exception as e:
        Logger.log("ADMIN DELETEUSER ERROR: Failed to delete user with ID '{}'; error: {}".format(userID, e))
        return JSONRes.ambiguousError()
    
    return JSONRes.new(200, "User deleted successfully.", ResType.SUCCESS)

@adminBP.route("/resetPassword", methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "userID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Target user ID required.", serialise=False)
    )
)
@checkSession(strict=True, provideUser=True, requireSuperuser=True)
def resetPassword(user: User):
    targetUserID: str = request.json.get("userID").strip()
    
    if targetUserID == user.id:
        return JSONRes.new(400, "Cannot reset your own password.", ResType.USERERROR)
    
    targetUser = None
    try:
        targetUser = User.load(targetUserID)
        if not isinstance(targetUser, User):
            return JSONRes.new(404, "User not found.", ResType.USERERROR)
    except Exception as e:
        Logger.log("ADMIN RESETPASSWORD ERROR: Failed to load user with ID '{}'; error: {}".format(targetUserID, e))
        return JSONRes.ambiguousError()
    
    newPassword: str = Universal.generateUniqueID(customLength=6)
    targetUser.pwd = Encryption.encodeToSHA256(newPassword)
    targetUser.authToken = None # log the user out if logged in already
    
    try:
        targetUser.save()
        targetUser.newLog("Admin Password Reset", "Changed to a random new password upon admin request.")
        Logger.log("ADMIN RESETPASSWORD: Password reset for user with ID '{}' and username '{}'.".format(targetUserID, targetUser.username))
    except Exception as e:
        Logger.log("ADMIN RESETPASSWORD ERROR: Failed to reset password for user with ID '{}'; error: {}".format(targetUserID, e))
        return JSONRes.ambiguousError()
    
    ThreadManager.defaultProcessor.addJob(EmailCentre.dispatch, AdminPasswordResetAlert(user=targetUser, newPwd=newPassword))
    
    return JSONRes.new(200, "Password reset successfully. An email has been sent to the user with the new password.", ResType.SUCCESS)

@adminBP.route("/invalidateUserSession", methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "userID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Target user ID required.", serialise=False)
    )
)
@checkSession(strict=True, provideUser=True, requireSuperuser=True)
def invalidateUserSession(user: User):
    targetUserID: str = request.json.get("userID").strip()
    
    if targetUserID == user.id:
        return JSONRes.new(400, "Cannot invalidate your own session.", ResType.USERERROR)
    
    targetUser = None
    try:
        targetUser = User.load(targetUserID)
        if not isinstance(targetUser, User):
            return JSONRes.new(404, "User not found.", ResType.USERERROR)
    except Exception as e:
        Logger.log("ADMIN INVALIDATEUSERSESSION ERROR: Failed to load user with ID '{}'; error: {}".format(targetUserID, e))
        return JSONRes.ambiguousError()
    
    targetUser.authToken = None
    try:
        targetUser.save()
        Logger.log("ADMIN INVALIDATEUSERSESSION: Invalidated session for user with ID '{}' and username '{}'.".format(targetUserID, targetUser.username))
    except Exception as e:
        Logger.log("ADMIN INVALIDATEUSERSESSION ERROR: Failed to invalidate session for user with ID '{}'; error: {}".format(targetUserID, e))
        return JSONRes.ambiguousError()
    
    return JSONRes.new(200, "User session invalidated successfully.", ResType.SUCCESS)