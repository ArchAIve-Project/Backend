import os, shutil, sys, json
from typing import List, Dict
from enum import Enum
from services import FileOps, Logger, Universal

class File:
    def __init__(self, filename: str, store: str, id: str=None, contentType: str=None, updated: str=None, updateMetadata: bool=False, forceExistence: bool=False):
        self.id = id
        self.filename = filename
        self.store = store
        self.contentType = contentType
        self.updated = updated
        self.updateMetadata = updateMetadata
        self.forceExistence = forceExistence
    
    def represent(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "store": self.store,
            "contentType": self.contentType,
            "updated": self.updated,
            "updateMetadata": self.updateMetadata,
            "forceExistence": self.forceExistence
        }
    
    def path(self):
        return os.path.join(os.getcwd(), self.store, self.filename)
    
    @staticmethod
    def from_dict(data: dict) -> 'File':
        return File(
            filename=data.get("filename"),
            store=data.get("store"),
            id=data.get("id"),
            contentType=data.get("contentType"),
            updated=data.get("updated"),
            updateMetadata=data.get("updateMetadata"),
            forceExistence=data.get("forceExistence")
        )
    
    def __str__(self):
        return f"File(name={self.filename}, store={self.store}, id={self.id if self.id != None else "None"}, contentType={self.contentType if self.contentType != None else "None"}, updated={self.updated if self.updated != None else "None"}, updateMetadata={self.updateMetadata if self.updateMetadata != None else "None"}, forceExistence={self.forceExistence if self.forceExistence != None else "None"})"

class FileManager:
    mode = "cloud"
    dataFile = "fmContext.json"
    stores = ["FileStore", "people", "artefacts"]
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
        # Check if a File object exists
        return FileOps.exists(file.path(), type="file")
    
    @staticmethod
    def delete(file: File):
        try:
            res = FileOps.deleteFile(file.path())
            if res != True:
                raise Exception(res)
        except Exception as e:
            return "ERROR: Failed to delete file; error: {}".format(e)
    
    @staticmethod
    def cleanupNonmatchingFiles():
        # Check non-matching files in all stores
        for store in FileManager.stores:
            filenames = FileOps.getFilenames(store)
            if isinstance(filenames, str):
                return "ERROR: Failed to get filenames for store '{}'; response: {}".format(filenames)
            
            for filename in filenames:
                if filename not in FileManager.context.keys():
                    FileOps.deleteFile(os.path.join(store, filename))
        
        return True
    
    @staticmethod
    def createStores(*args: list[str]):
        toCreate = FileManager.stores if len(args) == 0 else args
        
        for store in toCreate:
            res = FileOps.createFolder(store)
            
            if res != True:
                if store in FileManager.stores:
                    FileManager.stores.remove(store)
                return "ERROR: Failed to create store '{}'; error: {}".format(res)
            
            if store not in FileManager.stores:
                FileManager.stores.append(store)
        
        return True
    
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
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file with ID '{}' due to a value that is not of type 'dict'.".format(fileID))
            
            try:
                file = File.from_dict(data)
                if (file.filename is None) or (not isinstance(file.filename, str)) or (file.store is None) or (not isinstance(file.store, str)):
                    raise Exception("Filename or store is missing or invalid.")
                if not FileManager.exists(file):
                    raise Exception("File not found.")
                if file.store not in FileManager.stores:
                    raise Exception("Store not tracked by FileManager.")
                
                FileManager.context[fileID] = file
            except Exception as e:
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file with ID '{}' due to error: {}".format(fileID, e))
        
        FileManager.saveContext()
        FileManager.cleanupNonmatchingFiles()
        
        return True
    
    @staticmethod
    def saveContext() -> bool | str:
        try:
            with open(FileManager.dataFile, "w") as f:
                data = {file.id: file.represent() for file in FileManager.context.values()}
                data['mode'] = FileManager.mode
                
                json.dump(data, f)
        except Exception as e:
            return "ERROR: Failed to save context to file; error: {}".format(e)
        
        return True