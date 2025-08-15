from flask import request
from flask_limiter import Limiter

def getIP():
    return request.headers.get('X-Real-Ip', request.remote_addr)

limiter = Limiter(
    getIP,
    default_limits=["100 per minute"],
    storage_uri="memory://",
)