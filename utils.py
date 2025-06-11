from typing import List, Dict, Any
from flask import jsonify

class Ref:
    def __init__(self, *subscripts: tuple[str]) -> None:
        self.subscripts = list(subscripts)
        
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Ref):
            return False
        
        return self.subscripts == value.subscripts
        
    def __str__(self) -> str:
        return "/".join(self.subscripts)
    
class DIError:
    def __init__(self, message: str, supplementaryData: dict = None) -> None:
        self.message = message
        try:
            self.source = message[:message.index(":")]
        except:
            self.source = None
        self.supplementaryData = supplementaryData
        
    def __str__(self) -> str:
        return self.message

class ResType:
    success = "SUCCESS"
    error = "ERROR"
    userError = "UERROR"

class JSONRes:
    @staticmethod
    def new(code: int, msgType: str, msg: str, serialise: bool=True, serialiser: Any | None=None, **kwargs: Any) -> tuple:
        payload = {
            "type": msgType,
            "message": msg
        }
        
        if kwargs:
            payload.update(kwargs)
        
        if serialise and serialiser is None:
            serialiser = jsonify
        
        if serialiser:
            payload = serialiser(payload)
        
        return payload, code