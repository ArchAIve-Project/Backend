import os, functools, time
from types import FunctionType
from flask import request
from utils import JSONRes
from services import LiteStore

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

def databaseDebug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        if not os.environ.get("DB_DEBUG_MODE", "False") == "True":
            return func(*args, **kwargs)
        
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__}() returned {repr(value)}")
        return value
    return wrapper_debug

def timeit(func):
    """Print the time taken by the function to execute"""
    @functools.wraps(func)
    def wrapper_timeit(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
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

def cache(_func=None, *, ttl=60, lsInvalidator=None):
    """Cache the function's return value for a given time-to-live (TTL) or until it is invalidated by a `LiteStore` value being set to `True`."""
    def decorator_cache(func):
        @functools.wraps(func)
        @debug
        def wrapper_cache(*args, **kwargs):
            cache_key = args + tuple(kwargs.items())
            if cache_key in wrapper_cache.cache:
                _, timestamp = wrapper_cache.cache[cache_key]
                if (lsInvalidator and LiteStore.read(lsInvalidator) == True):
                    wrapper_cache.cache = {}
                    wrapper_cache.cache[cache_key] = (func(*args, **kwargs), time.time())
                    LiteStore.set(lsInvalidator, False)
                    
                    if os.environ.get("DECORATOR_DEBUG_MODE", "False") == "True":
                        print("CACHE DEBUG {}: Cache expired due to LS Invalidation for key: {}".format(func.__name__, cache_key))
                elif ttl and (time.time() - timestamp) > ttl:
                    wrapper_cache.cache = {}
                    wrapper_cache.cache[cache_key] = (func(*args, **kwargs), time.time())
                    
                    if os.environ.get("DECORATOR_DEBUG_MODE", "False") == "True":
                        print("CACHE DEBUG {}: Cache expired for key: {}".format(func.__name__, cache_key))
            else:
                wrapper_cache.cache[cache_key] = (func(*args, **kwargs), time.time())
            
            return wrapper_cache.cache[cache_key][0]
        
        wrapper_cache.cache = {}
        return wrapper_cache
    
    if _func is None:
        return decorator_cache
    else:
        return decorator_cache(_func)

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
        if ("API_KEY" in os.environ) and (("APIKey" not in request.headers) or (request.headers.get("APIKey", "") != os.environ.get("API_KEY", None))):
            return JSONRes.unauthorised()
        else:
            return func(*args, **kwargs)
    
    return wrapper_checkAPIKey

class Param:
    """Parameter for request validation.
    
    Each parameter has a name, a list of expectations (types, validation functions that return a `bool`, or specific values),
    and an optional invalid response (`invalidRes`).
    
    If the parameter is not present in the request, it will return `invalidRes`.
    
    If the parameter is present but does not match any of the expectations, it will return `invalidRes`.
    
    `invalidRes` defaults to `JSONRes.invalidRequestFormat()`.
    """
    
    def __init__(self, name: str, *expectations, invalidRes: JSONRes | tuple | str=None):
        """Initialize a new Param instance.

        Args:
            name (str): The name of the parameter.
            invalidRes (JSONRes | tuple | str, optional): The response to return if the parameter is invalid. Leave `None` to default as `JSONRes.invalidRequestFormat()`.
        """
        
        self.name = name
        self.invalidRes = invalidRes
        self.expectations = expectations
    
    def optional(self):
        return None in self.expectations
    
    def invalid(self):
        return self.invalidRes or JSONRes.invalidRequestFormat()
    
    @staticmethod
    def backwardsCompatible(value: tuple | str) -> 'Param':
        if isinstance(value, str):
            return Param(value)
        elif isinstance(value, tuple):
            if len(value) == 1:
                return Param(value[0])
            elif len(value) > 1:
                return Param(value[0], *value[1:])
        else:
            raise TypeError("PARAM BACKWARDSCOMPATIBLE ERROR: 'value' must be a string or a tuple.")

def enforceSchema(*expectedParams):
    """Enforce a specific schema for the JSON payload.
    
    Requires request to be in JSON format, or 415 Unsupported Media Type error will be raised.
    
    Sample usage:
    ```python
    from decorators import enforceSchema, Param
    
    @app.route('/example', methods=['POST'])
    @enforceSchema(
        Param("name", str), # Required, string expected
        Param("bmi", int, float), # Required, int or float expected
        Param("email", lambda x: isinstance(x, str) and '@' in x, None, invalidRes=JSONRes.new(400, "Invalid email format.", ResType.USERERROR)), # Optional, custom validation function and response
        Param("optionalField", str, None)  # Optional field        
    )
    def example():
        # Your function logic here
        return JSONRes.new(200, "Success")
    ```
    """
    def decorator_enforceSchema(func):
        @functools.wraps(func)
        @debug
        def wrapper_enforceSchema(*args, **kwargs):
            jsonData = request.get_json()
            
            for param in expectedParams:
                if not isinstance(param, Param):
                    # Convert param to Param if it is not already
                    param = Param.backwardsCompatible(param)
                
                # Check if the parameter is present in the JSON data
                if param.name not in jsonData:
                    if not param.optional():
                        return param.invalid()
                    else:
                        continue
                
                # Extract value and validate against expectations
                value = jsonData[param.name]
                if param.expectations:
                    valid = len(param.expectations) == 1 and param.expectations[0] is None
                    if valid:
                        # If only None is expected, any value is valid
                        continue
                    
                    for expect in param.expectations:
                        if isinstance(expect, type) and isinstance(value, expect):
                            # If the expectation is a type, check if value is of that type
                            valid = True
                            break
                        elif isinstance(expect, FunctionType) and expect(value):
                            # If the expectation is a function, call it with the value
                            valid = True
                            break
                        elif value == expect and expect is not None:
                            # If the expectation is a specific value, check if value matches it
                            valid = True
                            break
                    
                    if not valid:
                        # If none of the expectations matched, return invalid response
                        return param.invalid()
            
            return func(*args, **kwargs)
        
        return wrapper_enforceSchema
    
    return decorator_enforceSchema