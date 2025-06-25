import os, shutil, sys, json
from typing import List, Dict
from enum import Enum
from services import FileOps, Logger

class File:
    class Type(str, Enum):
        FACESHOT = "faceshot"
        ARTEFACT = "artefact"
        FACE_EMBEDDINGS = "embeddings"
        OTHERS = "others"
        
        def __eq__(self, value: 'File.Type'):
            return self.value == str(value)
        
        def __str__(self):
            return self.value
        
        @staticmethod
        def isValid(fileType: str):
            return fileType in [
                str(File.Type.FACESHOT),
                str(File.Type.ARTEFACT),
                str(File.Type.FACE_EMBEDDINGS),
                str(File.Type.OTHERS),
            ]
    
    def __init__(self, filename: str, fileType: 'File.Type | str', id: str=None, contentType: str=None, updated: str=None, updateMetadata: bool=False, forceExistence: bool=False):
        self.id = id
        self.filename = filename
        self.fileType = fileType.value if isinstance(fileType, File.Type) else fileType
        self.contentType = contentType
        self.updated = updated
        self.updateMetadata = updateMetadata
        self.forceExistence = forceExistence
    
    def represent(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "fileType": self.fileType,
            "contentType": self.contentType,
            "updated": self.updated,
            "updateMetadata": self.updateMetadata,
            "forceExistence": self.forceExistence
        }
    
    def path(self):
        if self.fileType == File.Type.FACESHOT:
            return os.path.join(os.getcwd(), "people", self.filename)
        elif self.fileType == File.Type.ARTEFACT:
            return os.path.join(os.getcwd(), "artefacts", self.filename)
        elif self.fileType == File.Type.FACE_EMBEDDINGS:
            return os.path.join(os.getcwd(), "embeddings.pth")
        else:
            return os.path.join(os.getcwd(), "FileStore", self.filename)
    
    @staticmethod
    def from_dict(data: dict) -> 'File':
        return File(
            filename=data.get("filename"),
            fileType=data.get("type"),
            id=data.get("id"),
            contentType=data.get("contentType"),
            updated=data.get("updated"),
            updateMetadata=data.get("updateMetadata"),
            forceExistence=data.get("forceExistence")
        )
    
    def __str__(self):
        return f"File(name={self.filename}, type={self.fileType}, id={self.id if self.id != None else "None"}, contentType={self.contentType if self.contentType != None else "None"}, updated={self.updated if self.updated != None else "None"}, updateMetadata={self.updateMetadata if self.updateMetadata != None else "None"}, forceExistence={self.forceExistence if self.forceExistence != None else "None"})"

class FileManager:
    mode = "cloud"
    dataFile = "fmContext.json"
    context: Dict[str, File] = {}
    
    @staticmethod
    def ensureDataFile():
        if not os.path.isfile(os.path.join(os.getcwd(), FileManager.dataFile)):
            with open(FileManager.dataFile, "w") as f:
                json.dump({
                    "mode": FileManager.mode
                }, f)
        
        return True
    
    @staticmethod
    def exists(file: File) -> bool:
        return os.path.isfile(file.path())
    
    @staticmethod
    def delete(file: File):
        try:
            os.remove(file.path())
        except Exception as e:
            return "ERROR: Failed to delete file; error: {}".format(e)
    
    @staticmethod
    def cleanupNonmatchingFiles():
        for file in FileManager.context.values():
            if not FileManager.exists(file):
                pass
    
    @staticmethod
    def loadContext() -> bool | str:
        FileManager.ensureDataFile()
        
        data: dict | None = None
        try:
            with open(FileManager.dataFile, "r") as f:
                data = json.load(f)
        except Exception as e:
            return "ERROR: Failed to load context data; error: {}".format(e)
        
        for fileID in data:
            if fileID == "mode": # TODO: context morphing for mode mismatch
                continue
            if not isinstance(data[fileID], dict):
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file ID '{}' due to a value that is not of type 'dict'.".format(fileID))
            
            try:
                file = File.from_dict(data)
                if (file.filename is None) or (not isinstance(file.filename, str)) or (file.fileType is None) or (not File.Type.isValid(file.fileType)):
                    raise Exception("Filename or type is missing or invalid.")
                if not FileManager.exists(file):
                    raise Exception("File not found.")
                
                FileManager.context[fileID] = file
            except Exception as e:
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file with ID '{}' due to error: {}".format(fileID, e))
        
        FileManager.saveContext()
        FileManager.cleanupNonmatchingFiles()
        
        return True
    
    @staticmethod
    def saveContext() -> bool | str:
        try:
            with open(FileManager.dataFile, "r") as f:
                data = {file.id: file.represent() for file in FileManager.context.values()}
                data['mode'] = FileManager.mode
                
                json.dump(data, f)
        except Exception as e:
            return "ERROR: Failed to save context to file; error: {}".format(e)
        
        return True