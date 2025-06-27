import os, shutil, sys, json, pprint
from typing import List, Dict
from enum import Enum
from services import FileOps, Logger, Universal
from firebase import FireStorage

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
    
    def identifierPath(self):
        return os.path.join(self.store, self.filename)
    
    def updateData(self, cloudData: dict):
        if not isinstance(cloudData, dict):
            return False
        
        try:
            if 'id' in cloudData:
                self.id = cloudData['id']
            if 'contentType' in cloudData:
                self.contentType = cloudData['contentType']
            if 'updated' in cloudData:
                self.updated = cloudData['updated'].isoformat()
            
            self.updateMetadata = False
            self.forceExistence = False
        except Exception as e:
            Logger.log("FILE UPDATEDATA ERROR: Failed to update for file '{}'; error: {}".format(self.identifierPath(), e))
        
        return True
    
    @staticmethod
    def generateIDPath(store: str, filename: str):
        return os.path.join(store, filename)
    
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
    initialised = False
    
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
    def debugPrint(msg: str) -> None:
        if os.environ.get("DEBUG_MODE") == "True":
            print("FM DEBUG: {}".format(msg))
    
    @staticmethod
    def delete(file: File):
        try:
            res = FileOps.deleteFile(file.identifierPath())
            if res != True:
                raise Exception(res)
            
            FileManager.debugPrint(f"'{file.identifierPath()}' deleted from system.")
            
            if file.identifierPath() in FileManager.context:
                del FileManager.context[file.identifierPath()]
                FileManager.saveContext()
                
                FileManager.debugPrint(f"'{file.identifierPath()}' deleted from context.")

            return True
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
                if File.generateIDPath(store, filename) not in FileManager.context.keys():
                    FileOps.deleteFile(os.path.join(store, filename))
                    
                    FileManager.debugPrint(f"'{File.generateIDPath(store, filename)}' deleted due to unmatched context reference.")
        
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
        
        # Read JSON data from data file
        data: dict | None = None
        try:
            with open(FileManager.dataFile, "r") as f:
                data = json.load(f)
        except Exception as e:
            return "ERROR: Failed to load context data; error: {}".format(e)
        
        # Iterate through all raw dictionary file information objects and instantiate File instances for in-memory use
        for fileIDPath in data:
            if fileIDPath == "mode": # TODO: context morphing for mode mismatch
                continue
            if not isinstance(data[fileIDPath], dict):
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file '{}' due to a value that is not of type 'dict'.".format(fileIDPath))
            
            try:
                file = File.from_dict(data[fileIDPath])
                if (file.filename is None) or (not isinstance(file.filename, str)) or (file.store is None) or (not isinstance(file.store, str)):
                    raise Exception("Filename or store is missing or invalid.")
                if not FileManager.exists(file):
                    raise Exception("File not found.")
                if file.store not in FileManager.stores:
                    raise Exception("Store not tracked by FileManager.")
                
                FileManager.context[file.identifierPath()] = file
                FileManager.debugPrint(f"'{file.identifierPath()}' loaded into context successfully.")
            except Exception as e:
                Logger.log("FILEMANAGER LOADCONTEXT WARNING: Skipping file '{}' due to error: {}".format(fileIDPath, e))
        
        # Examine files from all stores and add File instances for them to the context.
        # Warning: This means that File instances not loaded previously due to invalid data will be overwritten in the context.
        for store in FileManager.stores:
            filenames = FileOps.getFilenames(store)
            if isinstance(filenames, str):
                return "ERROR: Failed to obtain filenames for store '{}'; error: {}".format(store, filenames)
            
            for filename in filenames:
                if File.generateIDPath(store, filename) not in FileManager.context:
                    newFile = File(filename, store)
                    FileManager.context[newFile.identifierPath()] = newFile
                    
                    FileManager.debugPrint(f"'{newFile.identifierPath()}' added to context due to an existing file in the system.")
        
        FileManager.saveContext()
        FileManager.cleanupNonmatchingFiles()
        
        return True
    
    @staticmethod
    def saveContext() -> bool | str:
        try:
            with open(FileManager.dataFile, "w") as f:
                data = {file.identifierPath(): file.represent() for file in FileManager.context.values()}
                data['mode'] = FileManager.mode
                
                json.dump(data, f, indent=4)
        except Exception as e:
            return "ERROR: Failed to save context to file; error: {}".format(e)
        
        return True
    
    @staticmethod
    def setup():
        # Setup stores
        res = FileManager.createStores()
        if res != True:
            return "ERROR: Failed to create stores; error: {}".format(res)
        
        # Read, validate and load context into memory
        res = FileManager.loadContext()
        if res != True:
            return "ERROR: Failed to load context; error: {}".format(res)
        
        # Fetch information on cloud files
        cloudFiles = FireStorage.listFiles(ignoreFolders=True) # get blobs for all files only
        if isinstance(cloudFiles, str):
            return "ERROR: Failed to list files in cloud storage; error: {}".format(cloudFiles)
        cloudFiles = list(cloudFiles) # convert the HTTPIterator object from listFiles into a List[Blob]
        
        cloudFilenames = [b.name for b in cloudFiles] # for easy name reference. each name is like 'hello.txt', 'storeName/john.txt'
        cloudFiles = {b.name: FireStorage.getFileInfo(b.name, b) for b in cloudFiles} # extract name-keyed (corresponds with local File.identifierPath) cloud metadata information.
        
        for fileIDPath in list(FileManager.context.keys()):
            reDownloadOrMetadataUpdateAvailable = True
            
            # Check if file does not exist on cloud. If not, if 'forceExistence', upload, get metadata, and save; else, delete from context and fs.
            if fileIDPath not in cloudFilenames:
                if FileManager.context[fileIDPath].forceExistence:
                    # Force exist mandate on file context, upload to Firebase
                    res = FireStorage.uploadFile(
                        localFilePath=FileManager.context[fileIDPath].path(),
                        filename=fileIDPath
                    )
                    if res != True:
                        return "ERROR: Failed to upload force-existed file '{}'; error: {}".format(fileIDPath, res)
                    
                    # Get metadata on newly uploaded file
                    res = FireStorage.getFileInfo(fileIDPath)
                    if not isinstance(res, dict):
                        return "ERROR: Failed to obtain metadata for force-existed file '{}'; response: {}".format(fileIDPath, res)
                    
                    # Update context metadata with fetched cloud metadata
                    FileManager.context[fileIDPath].updateData(res)
                    
                    reDownloadOrMetadataUpdateAvailable = False # No need to re-download/update metadata as file was force existed
                    
                    FileManager.debugPrint(f"'{fileIDPath}': Force exist flag. Uploaded and updated metadata.")
                else:
                    # File not found and no force existence flag. Remove from filesystem and context
                    res = FileManager.delete(FileManager.context[fileIDPath])
                    if res != True:
                        return "ERROR: Failed to delete cloud-unmatched file '{}'; error: {}".format(res)
                    
                    reDownloadOrMetadataUpdateAvailable = False # No need to re-download/update metadata as file is deleted
                    
                    FileManager.debugPrint(f"'{fileIDPath}': File not found without force existence. Deleted from system and context.")
            else:
                FileManager.debugPrint(f"'{fileIDPath}': File found in cloud storage.")
            
            # Check if file needs to be redownloaded based on 'updated' parity. update metadata if so.
            if reDownloadOrMetadataUpdateAvailable:
                reDownloaded = False
                if FileManager.context[fileIDPath].updated != cloudFiles[fileIDPath]['updated'].isoformat():
                    # Local file needs to be updated with newer version available on cloud
                    
                    # Download the file
                    res = FireStorage.downloadFile(FileManager.context[fileIDPath].path(), fileIDPath)
                    if res != True:
                        return "ERROR: Failed to re-download out-of-date file '{}'; error: {}".format(res)

                    reDownloaded = True
                    FileManager.debugPrint(f"'{fileIDPath}': Redownloaded file to due to newer version available.")
                
                # If the file was redownloaded due to update parity or if the context has an updateMetadata mandate, update the context metadata
                if FileManager.context[fileIDPath].updateMetadata or reDownloaded:
                    FileManager.debugPrint(f"'{fileIDPath}': Context data updated due to updateMetadata: {FileManager.context[fileIDPath].updateMetadata} or reDownloaded: {reDownloaded}")
                    # Update context metadata
                    FileManager.context[fileIDPath].updateData(cloudFiles[fileIDPath])
        
        FileManager.saveContext()
        FileManager.cleanupNonmatchingFiles()
        FileManager.initialised = True
        
        return True

    @staticmethod
    def prepFile(store: str, filename: str) -> File | str:
        if not FileManager.initialised:
            return "ERROR: FileManager has not been setup yet."
        if store not in FileManager.stores:
            return "ERROR: Store '{}' is not tracked by FileManager.".format(store)
        
        idPath = File.generateIDPath(store, filename)
        
        cloudFile = FireStorage.getFileInfo(idPath)
        if cloudFile == None:
            # File does not exist. Delete from filesystem and context if it exists.
            fileExists = FileOps.exists(os.path.join(os.getcwd(), idPath), type="file")
            res = FileManager.delete(File(filename, store))
            if res != True:
                return "ERROR: Failed to delete file '{}' that does not exist on cloud storage; error: {}".format(idPath, res)
            
            if fileExists:
                FileManager.debugPrint(f"'{idPath}' deleted from system as it was not found on cloud during file prep.")
                FileManager.debugPrint(f"'{idPath}' deleted from context as it was not found on cloud during file prep.")
            
            return "ERROR: File does not exist."
        
        if not isinstance(cloudFile, dict):
            return "ERROR: Expected type 'dict' when retrieving cloud file information; response: {}".format(cloudFile)

        # File exists on cloud storage.
        
        ## Ensure existence in the context
        if idPath in FileManager.context:
            # File is in context
            
            if cloudFile['updated'].isoformat() != FileManager.context[idPath].updated:
                # File needs to be re-downloaded due to newer version
                res = FireStorage.downloadFile(FileManager.context[idPath].path(), idPath)
                if res != True:
                    return "ERROR: Failed to download newer version of file '{}' available on cloud storage; response: {}".format(idPath, res)
                
                FileManager.context[idPath].updateData(cloudFile)
                FileManager.saveContext()
                
                FileManager.debugPrint(f"'{idPath}' downloaded and updated in context as a newer version was found on cloud during file prep.")
        else:
            # File needs to be added to context
            newFile = File(filename, store)
            newFile.updateData(cloudFile)
            
            FileManager.context[newFile.identifierPath()] = newFile
            FileManager.saveContext()
            
            # File needs to be downloaded to ensure that the file is the same as cloud, otherwise it could a same filename but different data issue
            res = FireStorage.downloadFile(newFile.path(), newFile.identifierPath())
            if res != True:
                return "ERROR: Failed to re-download file '{}' from cloud storage for data integrity; error: {}".format(newFile.identifierPath(), res)
            
            FileManager.debugPrint(f"'{newFile.identifierPath()}' added to context and re-downloaded for data integrity as it was found on cloud during file prep.")
        
        ## Ensure existence in the filesystem
        if not FileManager.exists(FileManager.context[idPath]):
            # File is missing and needs to be downloaded
            res = FireStorage.downloadFile(FileManager.context[idPath].path(), idPath)
            if res != True:
                return "ERROR: Failed to download missing file '{}' from cloud storage; response: {}".format(idPath, res)
            
            FileManager.debugPrint(f"'{idPath}' downloaded as it was found on cloud during file prep.")
        
        return FileManager.context[idPath]

    @staticmethod
    def save(store: str, filename: str) -> File:
        if not FileManager.initialised:
            return "ERROR: FileManager has not been setup yet."
        if store not in FileManager.stores:
            return "ERROR: Store '{}' is not tracked by FileManager.".format(store)
        
        targetFile: File = None
        if File.generateIDPath(store, filename) in FileManager.context:
            targetFile = FileManager.context[File.generateIDPath(store, filename)]
        else:
            targetFile = File(filename, store)
        
        if not FileManager.exists(targetFile):
            return "ERROR: File '{}' does not exist.".format(targetFile.identifierPath())
        
        res = FireStorage.uploadFile(targetFile.path(), targetFile.identifierPath())
        if res != True:
            return "ERROR: Failed to upload file '{}' to cloud storage; response: {}".format(targetFile.identifierPath(), res)
        
        fileInfo = FireStorage.getFileInfo(targetFile.identifierPath())
        if not isinstance(fileInfo, dict):
            return "ERROR: Expected type 'dict' when retrieving information on uploaded file; response: {}".format(fileInfo)
        
        targetFile.updateData(fileInfo)
        
        FileManager.context[targetFile.identifierPath()] = targetFile
        FileManager.saveContext()
        
        return targetFile