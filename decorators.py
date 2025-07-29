import os, functools, time
from types import FunctionType
from flask import request
from utils import JSONRes

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
        if ("API_KEY" in os.environ) and (("APIKey" not in request.headers) or (request.headers.get("APIKey", "") != os.environ.get("API_KEY", None))):
            return JSONRes.unauthorised()
        else:
            return func(*args, **kwargs)
    
    return wrapper_checkAPIKey

def enforceSchema(*expectedArgs):
    """Enforce a specific schema for the JSON payload.
    
    Requires request to be in JSON format, or 415 Unsupported Media Type error will be raised.
    
    Sample implementation:
    ```python
    @app.route("/")
    @enforceSchema(
        ("hello", int),
        "world",
        ("john"),
        ("smth", str, "specificValue", lambda x: isinstance(x, str) and len(x) > 5, None)
    )
    def home():
        return "Success!"
    ```
    
    The above example enforces that:
    - `hello` is provided and is a valid `int`.
    - `world` is provided and can be of any type.
    - `john` is provided and can be of any type.
    - `smth` is optionally provided, and should it exist, it should be either a `str`, have the value of `specificValue`, or be a `str` with length greater than 5
    
    The validation schema is quite flexible. For every parameter in `expectedArgs`:
    - If only the name is provided, either as a string or as the only item in a tuple, it just checks that the parameter is present in the JSON payload.
    - In a tuple, the first element is considered as the parameter name, and the rest are considered as expectations for that parameter.
    - To make a parameter optional, include `None` in the expectations.
    - Do note that the parameter only has to match any ONE expectation to be considered valid.
    - Expectations can be:
        - A type (like `int`, `str`, etc.) to ensure the parameter is of that type.
        - A function that takes the parameter value and returns `True` if the value is valid, and `False` otherwise.
        - A specific value to ensure the parameter matches that value.
        - `None` to indicate optionality.
    
    Returns `JSONRes.invalidRequestFormat` if the schema is not met.
    """
    def decorator_enforceSchema(func):
        @functools.wraps(func)
        @debug
        def wrapper_enforceSchema(*args, **kwargs):
            jsonData = request.get_json()
            
            for param in expectedArgs:
                if isinstance(param, tuple):
                    key, *expectations = param
                    value_present = key in jsonData
                    
                    if value_present:
                        value = jsonData[key]
                        if expectations:
                            valid = len(expectations) == 1 and expectations[0] is None
                            if valid:
                                # If only None is expected, any value is valid
                                continue
                            
                            for expect in expectations:
                                if isinstance(expect, type) and isinstance(value, expect):
                                    valid = True
                                    break
                                elif isinstance(expect, FunctionType) and expect(value):
                                    valid = True
                                    break
                                elif value == expect and expect is not None:
                                    valid = True
                                    break
                            
                            if not valid:
                                return JSONRes.invalidRequestFormat()
                    elif None not in expectations:
                        return JSONRes.invalidRequestFormat()
                elif param not in jsonData:
                    return JSONRes.invalidRequestFormat()
            
            return func(*args, **kwargs)
        
        return wrapper_enforceSchema
    
    return decorator_enforceSchema