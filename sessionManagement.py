import os, functools
from decorators import debug
from utils import JSONRes
from services import Logger, Universal
from flask import session
from schemas import User

class CheckSessionOutput:
    def __init__(self, reason: str, outputValue) -> None:
        self.reason = reason
        self.outputValue = outputValue

def sessionCheckDebug(func):
    """Examine how the session was checked."""
    @functools.wraps(func)
    def wrapper_sessionCheckDebug(*args, **kwargs):
        res: CheckSessionOutput = func(*args, **kwargs)
        if os.environ.get("DECORATOR_DEBUG_MODE", "False") == "True":
            print("CHECKSESSION DEBUG: Return: {}, Reason: {}".format(res.outputValue, res.reason))
        return res.outputValue
    
    return wrapper_sessionCheckDebug


def checkSession(_func=None, *, strict=False, provideUser=False):
    def decorator_checkSession(func):
        @functools.wraps(func)
        @debug
        @sessionCheckDebug
        def wrapper_checkSession(*args, **kwargs):
            def handleReturn(res: tuple, reason: str=None, user: User | None=None, clearSessionForErrorCodes: bool = True):
                code = res[1]
                if code != 200 and clearSessionForErrorCodes and len(session.keys()) != 0:
                    session.clear()
                
                if code != 200 and strict:
                    return CheckSessionOutput(reason, res)
                elif provideUser and user != None:
                    return CheckSessionOutput(reason, func(user, *args, **kwargs))
                else:
                    return CheckSessionOutput(reason, func(*args, **kwargs))

            reqParams = ["accID", "username", "authToken", "sessionStart", "superuser"]
            
            # Check if session parameters match up
            if len(session.keys()) != len(reqParams):
                return handleReturn(res=JSONRes.invalidSession(), reason="Session parameter count mismatch.")
            
            # Check if session is corrupted
            for param in reqParams:
                if param not in session:
                    return handleReturn(res=JSONRes.invalidSession(), reason="Session parameter '{}' missing.".format(param))
            
            # If strict mode, verify the session by trying to load the user based on it
            user = None
            if strict or provideUser:
                try:
                    # Load the user by passing in the account ID from the session
                    user = User.load(id=session["accID"])
                    
                    if not isinstance(user, User):
                        # If the user was not found, reset the session and return an error.
                        return handleReturn(res=JSONRes.invalidSession(), reason="User not found.")
                    
                    if user.authToken != session["authToken"]:
                        # If the authToken does not match, reset the session and return an error.
                        return handleReturn(res=JSONRes.invalidSession(), reason="Auth token mismatch.")
                except Exception as e:
                    Logger.log("CHECKSESSION ERROR: Failed to load user for strict auth token verification. Error: {}".format(e))
                    return handleReturn(res=JSONRes.invalidSession(), reason="Exception in user loading: {}".format(e))
                    
                try:
                    # Check the session start time for expiration cases
                    if not isinstance(user.lastLogin, str):
                        return handleReturn(res=JSONRes.invalidSession(), reason="Invalid lastLogin.")
                    
                    if (Universal.utcNow() - Universal.fromUTC(user.lastLogin)).total_seconds() > 7200:
                        ## Session expired
                        user.authToken = None
                        user.save()
                        return handleReturn(
                            res=JSONRes.new(401, "Session expired."),
                            reason="Session expired."
                        )
                except Exception as e:
                    Logger.log("CHECKSESSION ERROR: Failed to process session expiration status. Error: {}".format(e))
                    return handleReturn(res=JSONRes.invalidSession(), reason="Exception in session expiration: {}.".format(e))
            
            # Let handleReturn handle the success case
            return handleReturn(res=JSONRes.new(200, "Session verified."), reason="Session valid.", user=user)
        
        return wrapper_checkSession
    
    if _func is None:
        return decorator_checkSession
    else:
        return decorator_checkSession(_func)

def requireSuperuser(func):
    """Examine how the session was checked."""
    @functools.wraps(func)
    @debug
    def wrapper_requireSuperuser(user: User, *args, **kwargs):
        if not isinstance(user, User):
            Logger.log("REQUIRESUPERUSER WARNING: 'user' is not a valid User object. Returning unauthorised response as contingency.")
            return JSONRes.unauthorised()
        
        if not user.superuser:
            return JSONRes.unauthorised()
        
        return func(user, *args, **kwargs)
    
    return wrapper_requireSuperuser