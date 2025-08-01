from typing import List, Dict, Any
from enum import Enum
from pympler.asizeof import asizeof as sizeOf
from flask import jsonify

class Memory:
    @staticmethod
    def getSize(anyObject):
        return sizeOf(anyObject)

class Extensions:
    @staticmethod
    def sanitiseData(data: dict, allowedKeys: List[str]=None, disallowedKeys: List[str]=None, allowedTopLevelKeys: List[str]=None):
        if allowedKeys is None:
            allowedKeys = []
        if disallowedKeys is None:
            disallowedKeys = []
        if allowedTopLevelKeys is None:
            allowedTopLevelKeys = []
        
        if not isinstance(data, dict):
            return data

        outputDict = {}
        for key, value in data.items():
            if key in allowedTopLevelKeys:
                outputDict[key] = value
                continue
            
            if len(allowedKeys) > 0 and key not in allowedKeys:
                continue
            if len(disallowedKeys) > 0 and key in disallowedKeys:
                continue
            
            if isinstance(value, dict):
                outputDict[key] = Extensions.sanitiseData(value, allowedKeys, disallowedKeys)
            elif isinstance(value, list):
                outputDict[key] = [Extensions.sanitiseData(item, allowedKeys, disallowedKeys) for item in value]
            else:
                outputDict[key] = value
        
        return outputDict

class Ref:
    illegalChars = ['.', '#', '$', '[', ']', '/']
    
    def __init__(self, *subscripts: tuple[str]) -> None:
        self.subscripts = list(subscripts)
    
    def hasIllegalChars(self) -> bool:
        for sub in self.subscripts:
            for char in sub:
                if char in Ref.illegalChars:
                    return True
        
        return False
    
    def removeIllegalChars(self) -> bool:
        removed = False
        for subIndex in range(len(self.subscripts)):
            sub: str = self.subscripts[subIndex]
            for char in Ref.illegalChars:
                if char in sub:
                    sub = sub.replace(char, "")
                    removed = True
            
            self.subscripts[subIndex] = sub
        
        return removed
    
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

class ResType(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    USERERROR = "UERROR"
    
    @staticmethod
    def guess(code: int) -> 'ResType':
        if code >= 200 and code < 300:
            return ResType.SUCCESS
        else:
            return ResType.ERROR

class JSONRes:
    @staticmethod
    def new(code: int, msg: str, msgType: ResType | str=None, serialise: bool=True, serialiser: Any | None=None, **kwargs: Any) -> tuple:
        if msgType is None or (not isinstance(msgType, ResType) and not isinstance(msgType, str)):
            msgType = ResType.guess(code)
        
        if isinstance(msgType, ResType):
            msgType = msgType.value
        
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
    
    @staticmethod
    def invalidRequestFormat(msg: str = "Invalid request format.", **kwargs) -> tuple:
        return JSONRes.new(400, msg, ResType.ERROR, **kwargs)
    
    @staticmethod
    def unauthorised(msg: str="Request unauthorised.", **kwargs) -> tuple:
        return JSONRes.new(401, msg, ResType.ERROR, **kwargs)
    
    @staticmethod
    def invalidSession(msg: str="Invalid session.", **kwargs) -> tuple:
        return JSONRes.new(401, msg, ResType.ERROR, **kwargs)
    
    @staticmethod
    def ambiguousError(msg: str="Failed to process request. Please try again.", **kwargs) -> tuple:
        return JSONRes.new(500, msg, ResType.ERROR, **kwargs)