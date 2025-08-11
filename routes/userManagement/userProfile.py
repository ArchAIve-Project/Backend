import re
from flask import Blueprint, url_for, request, redirect
from utils import JSONRes, ResType
from services import Universal, Logger, Encryption, FileOps, ThreadManager
from decorators import jsonOnly, enforceSchema, checkAPIKey, Param
from sessionManagement import checkSession
from fm import File, FileManager
from schemas import User, AuditLog
from emailCentre import EmailCentre, PasswordChangedAlert

profileBP = Blueprint('profile', __name__, url_prefix='/profile')

@profileBP.route('/info', methods=['GET'])
@checkSession(strict=True, provideUser=True)
def info(user: User):
    return redirect(url_for('cdn.getProfileInfo', userID=user.id, includeLogs=request.args.get('includeLogs', 'false')))

@profileBP.route('/update', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "username",
        lambda x: isinstance(x, str) and len(x) >= 3 and x.isalnum() and len(x) <= 12, None,
        invalidRes=JSONRes.new(400, "Username must be between 3 and 12 characters, without any spaces or special characters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "fname",
        lambda x: isinstance(x, str) and len(x) >= 2 and x.replace(' ', '').isalpha() and len(x) <= 15, None,
        invalidRes=JSONRes.new(400, "First name must be between 2 and 15 characters and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "lname",
        lambda x: isinstance(x, str) and len(x) >= 1 and x.replace(' ', '').isalpha() and len(x) <= 15, None,
        invalidRes=JSONRes.new(400, "Last name must be between 1 and 15 characters and have only letters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "contact",
        lambda x: isinstance(x, str) and len(x.replace(' ', '').strip()) == 8 and x.replace(' ', '').strip().isdigit(), None,
        invalidRes=JSONRes.new(400, "Contact number must be 8 digits.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "email",
        lambda x: isinstance(x, str) and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x.strip()), None,
        invalidRes=JSONRes.new(400, "Invalid email.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "role",
        lambda x: isinstance(x, str) and len(x.strip()) >= 2 and len(x.strip()) <= 20, None,
        invalidRes=JSONRes.new(400, "Role must be between 2 and 20 characters.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "userID",
        str, None,
        invalidRes=JSONRes.new(400, "User ID must be a valid string if provided.", ResType.ERROR, serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
def update(user: User):
    superuserPrivilege = user.superuser == True
    
    # Carry out authorisation check. Superusers can update any user, others can only update their own profile.
    userID: str = request.json.get('userID', None)
    if userID and userID != user.id:
        if not user.superuser:
            # User is not the same as the requested user and is not a superuser
            return JSONRes.unauthorised()
        else:
            # User is a superuser, so we can load the requested user
            try:
                user = User.load(id=userID)
                if user is None:
                    return JSONRes.new(404, "User not found.")
                if not isinstance(user, User):
                    raise Exception("Unexpected load response: {}".format(user))
            except Exception as e:
                Logger.log("USERPROFILE UPDATE ERROR: Failed to load user '{}' for superuser request: {}".format(userID, e))
                return JSONRes.ambiguousError()
    
    username = request.json.get('username', user.username).strip()
    fname = request.json.get('fname', user.fname).strip()
    lname = request.json.get('lname', user.lname).strip()
    contact = request.json.get('contact', user.contact).replace(' ', '').strip()
    email = request.json.get('email', user.email).strip()
    role = request.json.get('role', user.role).strip()
    
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
    if role != user.role and superuserPrivilege:
        user.role = role
        changes.append("Role")
    
    if changes:
        try:
            user.save()
        except Exception as e:
            # Possible key violation
            e = str(e)
            if e.startswith("USER SAVE ERROR: Integrity violation: "):
                violation = e[len("USER SAVE ERROR: Integrity violation: "):]
                if violation == "Username":
                    return JSONRes.new(400, "Username already exists.", ResType.USERERROR)
                elif violation == "Email":
                    return JSONRes.new(400, "Email already exists.", ResType.USERERROR)
            
            Logger.log("USERPROFILE UPDATE ERROR: Failed to save user '{}' after updating details; error: {}".format(user.id, e))
            return JSONRes.ambiguousError()

        user.newLog("Profile Update", "{} details updated.{}".format(", ".join(changes), " (Admin)" if superuserPrivilege and user.superuser != True else ""))
    else:
        return JSONRes.new(200, "No changes made to the profile.")
    
    return JSONRes.new(200, "Profile updated successfully.")

@profileBP.route('/changePassword', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "currentPassword",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Current password is required.", ResType.USERERROR, serialise=False)
    ),
    Param(
        "newPassword",
        lambda x: isinstance(x, str) and len(x) >= 6 and len(x) <= 16 and (not re.search(r'\s', x)),
        invalidRes=JSONRes.new(400, "New password must be between 6 and 16 characters and contain no spaces.", ResType.USERERROR, serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
def changePassword(user: User):
    currentPassword = request.json['currentPassword'].strip()
    newPassword = request.json['newPassword'].strip()
    
    if newPassword == currentPassword:
        return JSONRes.new(400, "New password cannot be same as current password.", ResType.USERERROR)
    
    if not Encryption.verifySHA256(currentPassword, user.pwd):
        return JSONRes.new(401, "Current password is incorrect.", ResType.USERERROR)
    
    user.pwd = Encryption.encodeToSHA256(newPassword)
    
    try:
        user.save()
        user.newLog("Password Change", "Password changed successfully.")
    except Exception as e:
        Logger.log("USERPROFILE CHANGEPASSWORD ERROR: Failed to save user '{}' after changing password; error: {}".format(user.id, e))
        return JSONRes.ambiguousError()
    
    ThreadManager.defaultProcessor.addJob(EmailCentre.dispatch, PasswordChangedAlert(user))
    
    return JSONRes.new(200, "Password changed successfully.")

@profileBP.route('/uploadPicture', methods=['POST'])
@checkAPIKey
@checkSession(strict=True, provideUser=True)
def uploadPicture(user: User):
    file = request.files.get('file')
    if not file:
        return JSONRes.new(400, "No file part in the request.")
    if file.filename == '':
        return JSONRes.new(400, "No image selected.", ResType.USERERROR)
    
    # Check if file is a valid image
    filename = file.filename
    if not FileOps.allowedFileExtension(filename):
        return JSONRes.new(400, "Only valid image files are allowed (png, jpg, jpeg).", ResType.USERERROR)
    
    # Check file size (optional, e.g., max 5MB)
    MAX_FILE_SIZE = 5 * 1024 * 1024
    if FileOps.getFileStorageSize(file) > MAX_FILE_SIZE:
        return JSONRes.new(400, "File size exceeds 5MB limit.", ResType.USERERROR)
    
    # Delete existing profile picture if it exists
    if user.pfp:
        try:
            prevFile = File(user.pfp, 'FileStore')
            
            res = FileManager.delete(file=prevFile)
            if res != True:
                raise Exception(res)
        except Exception as e:
            Logger.log("USERPROFILE UPLOADPICTURE ERROR: Failed to delete existing profile picture for user '{}'; error: {}".format(user.id, e))
    
    user.pfp = None
    
    # Save new profile picture
    try:
        newFilename = '{}.{}'.format(user.id, FileOps.getFileExtension(filename))
        newFile = File(newFilename, 'FileStore')
        
        file.save(newFile.path())
        
        res = FileManager.save(file=newFile)
        if isinstance(res, str):
            raise Exception(res)
    except Exception as e:
        Logger.log("USERPROFILE UPLOADPICTURE ERROR: Failed to save new profile picture for user '{}'; error: {}".format(user.id, e))
        user.save() # Save user without profile picture
        return JSONRes.ambiguousError()
    
    # Offload new profile picture
    res = FileManager.offload(file=newFile)
    if isinstance(res, str):
        Logger.log("USERPROFILE UPLOADPICTURE WARNING: Failed to offload new profile picture for user '{}'; error: {}".format(user.id, res))
    
    # Save the new profile picture filename to the user
    user.pfp = newFilename
    
    try:
        user.save()
    except Exception as e:
        Logger.log("USERPROFILE UPLOADPICTURE ERROR: Failed to save user '{}' after uploading profile picture; error: {}".format(user.id, e))
        return JSONRes.ambiguousError()
    
    user.newLog("Profile Picture Updated", "Profile picture updated successfully.")
    
    return JSONRes.new(200, "Profile picture updated successfully.")

@profileBP.route('/deletePicture', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "userID",
        lambda x: isinstance(x, str) and len(x.strip()) > 0, None,
        invalidRes=JSONRes.new(400, "User ID must be a valid string, if provided.", serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
def deletePicture(user: User):
    targetUser: User = None
    targetUserID: str = request.json.get("userID", user.id).strip()
    
    # Resolve target user
    if targetUserID == user.id:
        targetUser = user
    elif user.superuser:
        try:
            targetUser = User.load(targetUserID)
            if not isinstance(targetUser, User):
                return JSONRes.new(404, "User not found.", ResType.USERERROR)
        except Exception as e:
            Logger.log("USERPROFILE DELETEPICTURE ERROR: Failed to load user '{}' for superuser request; error: {}".format(targetUserID, e))
            return JSONRes.ambiguousError()
    else:
        return JSONRes.unauthorised()
    
    if not targetUser.pfp:
        return JSONRes.new(200, "No profile picture to delete.")
    
    file = File(targetUser.pfp, 'FileStore')
    res = FileManager.delete(file=file)
    if isinstance(res, str):
        Logger.log("USERPROFILE DELETEPICTURE ERROR: Failed to delete profile picture for user '{}' (ID: {}); error: {}".format(targetUser.username, targetUser.id, res))
        return JSONRes.ambiguousError()
    
    targetUser.pfp = None
    try:
        targetUser.save()
        if user.superuser and targetUserID != user.id:
            targetUser.newLog("Profile Picture Deleted", "An admin deleted this account's profile picture.")
            Logger.log("USERPROFILE DELETEPICTURE: Admin deleted profile picture for user '{}' (ID: {}).".format(targetUser.username, targetUser.id))
    except Exception as e:
        Logger.log("USERPROFILE DELETEPICTURE ERROR: Failed to delete profile picture for user '{}'; error: {}".format(targetUser.id, e))
        return JSONRes.ambiguousError()
    
    return JSONRes.new(200, "Profile picture deleted successfully.")