import os, functools, time
from flask import request, session
from models import User
from services import Logger, Universal
from utils import JSONRes, ResType

class CheckSessionOutput:
    def __init__(self, reason: str, outputValue) -> None:
        self.reason = reason
        self.outputValue = outputValue

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        if not os.environ.get("DECORATOR_DEBUG_MODE", "False") == "True":
            return func(*args, **kwargs)
        
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__}() returned {repr(value)}")
        return value
    return wrapper_debug

def sessionCheckDebug(func):
    """Examine how the session was checked."""
    @functools.wraps(func)
    def wrapper_sessionCheckDebug(*args, **kwargs):
        res: CheckSessionOutput = func(*args, **kwargs)
        if os.environ.get("DECORATOR_DEBUG_MODE", "False") == "True":
            print("CHECKSESSION DEBUG: Return: {}, Reason: {}".format(res.outputValue, res.reason))
        return res.outputValue
    
    return wrapper_sessionCheckDebug

def timeit(func):
    """Print the time taken by the function to execute"""
    @functools.wraps(func)
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return value
    return wrapper_timeit

def slow_down(_func=None, *, rate=1):
    """Sleep given amount of seconds before calling the function"""
    def decorator_slow_down(func):
        @functools.wraps(func)
        def wrapper_slow_down(*args, **kwargs):
            time.sleep(rate)
            return func(*args, **kwargs)
        return wrapper_slow_down

    if _func is None:
        return decorator_slow_down
    else:
        return decorator_slow_down(_func)

def cache(func):
    """Keep a cache of previous function calls"""
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = {}
    return wrapper_cache

def jsonOnly(func):
    """Enforce the request to be in JSON format. 400 error if otherwise."""
    @functools.wraps(func)
    @debug
    def wrapper_jsonOnly(*args, **kwargs):
        if not request.is_json:
            return JSONRes.invalidRequestFormat()
        else:
            return func(*args, **kwargs)
    
    return wrapper_jsonOnly

def checkAPIKey(func):
    """Check if the request has the correct API key. 401 error if otherwise."""
    @functools.wraps(func)
    @debug
    def wrapper_checkAPIKey(*args, **kwargs):
        if ("API_KEY" in os.environ) and (("API_KEY" not in request.headers) or (request.headers["API_KEY"] != os.environ.get("API_KEY", None))):
            return JSONRes.unauthorised()
        else:
            return func(*args, **kwargs)
    
    return wrapper_checkAPIKey

def enforceSchema(*expectedArgs):
    """Enforce a specific schema for the JSON payload.
    
    Requires request to be in JSON format, or 415 werkzeug.exceptions.BadRequest error will be raised.
    
    Sample implementation:
    ```python
    @app.route("/")
    @enforceSchema(("hello", int), "world", ("john"))
    def home():
        return "Success!"
    ```
    
    The above implementation mandates that 'hello', 'world' and 'john' attributes must be present. Additionally, 'hello' must be an integer.
    
    Parameter definition can be one of the following:
    - String: Just ensures that the parameter is present, regardless of its value's datatype.
    - Tuple, with two elements: First element is the parameter name, second element is the expected datatype.
        - If the datatype (second element) is not provided, it will be as if the parameter's name was provided directly, i.e. it will just ensure that the parameter is present.
        - Example: `("hello", int)` ensures that 'hello' is present and is an integer. But, `("hello")` ensures that 'hello' is present, but does not enforce any datatype.
        
    Raises `"ERROR: Invalid request format", 400` if the schema is not met.
    """
    def decorator_enforceSchema(func):
        @functools.wraps(func)
        @debug
        def wrapper_enforceSchema(*args, **kwargs):
            jsonData = request.get_json()
            for expectedTuple in expectedArgs:
                if isinstance(expectedTuple, tuple):
                    if expectedTuple[0] not in jsonData:
                        return JSONRes.invalidRequestFormat()
                    if len(expectedTuple) > 1:
                        if not isinstance(jsonData[expectedTuple[0]], expectedTuple[1]):
                            return JSONRes.invalidRequestFormat()
                elif expectedTuple not in jsonData:
                    return JSONRes.invalidRequestFormat()
            
            return func(*args, **kwargs)
        
        return wrapper_enforceSchema
    
    return decorator_enforceSchema

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