import os, shutil, sys, json, pprint, datetime, threading
from typing import List, Dict
from enum import Enum
from services import FileOps, Logger, Universal
from firebase import FireStorage

class File:
    """
    Represents a file that is managed and synchronised with Firebase Cloud Storage.

    The File class is primarily used to represent a file by its metadata and path. It can
    be used independently of the local filesystem (e.g., to get signed URLs, fetch cloud metadata).

    Example 1:
    ```python
    f = File("image.jpg", "artefacts")
    if f.exists()[0]:
        print(f.getSignedURL())
        print(f.getPublicURL())
        print(f.disablePublicURL())
        print(f.path())
    ```
    
    Example 2 (let's say the file is on cloud and you want to retrive it's information):
    ```python
    f = File("image.jpg", "artefacts")
    f.fetchData()
    print(f)
    ```
    
    Usage with `FileManager`:
    
    Example 1:
    ```python
    FileManager.setup()
    
    with open("myStore/file.txt", "w") as f:
        f.write("Hello, world!")
    
    file = FileManager.save("myStore", "file.txt")
    print(file.getSignedURL())
    ```
    
    Example 2 (let's say a file is available on cloud but not locally):
    ```python
    FileManager.setup()
    
    f = FileManager.prepFile("myStore", "file.txt") # downloads the file into local context and system
    if isinstance(f, str):
        print("Something went wrong:", f)
    
    print(f.getSignedURL())
    print(f.getBytes())
    ```
    """
    
    def __init__(self, filename: str, store: str, id: str=None, contentType: str=None, updated: str=None, updateMetadata: bool=False, forceExistence: bool=False):
        """
        Initializes a File instance with optional cloud metadata and flags.

        Args:
            filename (str): Name of the file.
            store (str): Store/folder the file belongs to.
            id (str, optional): Firebase file ID.
            contentType (str, optional): MIME type.
            updated (str, optional): ISO timestamp of last update.
            updateMetadata (bool, optional): Force metadata update.
            forceExistence (bool, optional): Mandate cloud existence on setup.
        """ 
        
        self.id = id
        self.filename = filename
        self.store = store
        self.contentType = contentType
        self.updated = updated
        self.updateMetadata = updateMetadata
        self.forceExistence = forceExistence
    
    def filenameWithoutExtension(self) -> str:
        """Returns the filename without its extension.

        Returns:
            str: The filename without the extension.
        """
        return os.path.splitext(self.filename)[0]
    
    def represent(self) -> dict:
        """
        Returns the metadata of the file as a dictionary.

        Returns:
            dict: Dictionary representation of the file.
        """
        
        return {
            "id": self.id,
            "filename": self.filename,
            "store": self.store,
            "contentType": self.contentType,
            "updated": self.updated,
            "updateMetadata": self.updateMetadata,
            "forceExistence": self.forceExistence
        }
    
    def path(self) -> str:
        """
        Returns the local filesystem path of the file.

        Returns:
            str: Absolute local path.
        """
        
        return os.path.join(os.getcwd(), self.store, self.filename)
    
    def identifierPath(self) -> str:
        """
        Returns the unique Firebase identifier path ("store/filename").

        Returns:
            str: Identifier path.
        """
        
        return f"{self.store}/{self.filename}"
    
    def updateData(self, cloudData: dict) -> bool:
        """
        Updates metadata using the data retrieved from Firebase.

        Args:
            cloudData (dict): Metadata dictionary from Firebase.

        Returns:
            bool: True if updated successfully, False otherwise.
        """
        
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
    
    def exists(self):
        """
        Checks if the file exists in Firebase and the in-memory context. Uses `FileManager.exists` internally.

        Returns:
             Tuple[bool, bool] | str: (cloud_exists, context_exists) or error message.
        """
        
        return FileManager.exists(self)
    
    def fetchData(self) -> bool | str:
        """
        Fetches the latest metadata from Firebase and updates the self in-place.

        Returns:
             bool | str: True if successful, else error string.
        """
        
        res = FireStorage.getFileInfo(self.identifierPath())
        if isinstance(res, str):
            return "ERROR: Failed to get file information from cloud storage; response: {}".format(res)
        if res is None:
            return "ERROR: File does not exist."
        if not isinstance(res, dict):
            return "ERROR: Expected type 'dict' in file information retrieval; response: {}".format(res)
        
        self.updateData(res)
        return True
    
    def getPublicURL(self) -> str:
        """
        Makes the file ***public*** and retrieves its public URL. An object's public URL does not change. Use this with caution.
        
        Using signed URLs is recommended instead.

        Returns:
            str: Public URL or error message.
        """
        
        exists = self.exists()
        if isinstance(exists, str):
            return exists
        exists = exists[0]
        if not exists:
            return "ERROR: File does not exist."
        
        res = FireStorage.changeFileACL(self.identifierPath(), private=False)
        if res != True:
            return "ERROR: Failed to enable public access; response: {}".format(res)
        
        res = FireStorage.getFilePublicURL(self.identifierPath())
        if res.startswith("ERROR"):
            return "ERROR: Failed to retrieved public URL; response: {}".format(res)
        
        return res
    
    def disablePublicURL(self) -> bool | str:
        """
        Revokes public access to the file in Firebase. Public URLs will not work.

        Returns:
             bool | str: True if successful, else error message.
        """
        
        exists = self.exists()
        if isinstance(exists, str):
            return exists
        exists = exists[0]
        if not exists:
            return "ERROR: File does not exist."
        
        res = FireStorage.changeFileACL(self.identifierPath(), private=True)
        if res != True:
            return "ERROR: Failed to disable public access; response: {}".format(res)
        
        return True
    
    def getSignedURL(self, expiration: datetime.timedelta=None, useCache: bool=True) -> str:
        """
        Retrieves a signed (private) URL for the file. This URL cannot be revoked and will only stop working when it expires.

        Args:
            expiration (datetime.timedelta, optional): Time before URL expiry from current time.
            useCache (bool, optional): Use cached version if available.

        Returns:
            str: Signed URL or error message.
        """
        
        exists = self.exists()
        if isinstance(exists, str):
            return exists
        exists = exists[0]
        if not exists:
            return "ERROR: File does not exist."
        
        res = FireStorage.getFileSignedURL(self.identifierPath(), expiration=expiration, allowCache=useCache, updateCache=useCache)
        if res.startswith("ERROR"):
            return "ERROR: Failed to get signed URL; response: {}".format(res)
        
        return res
    
    def getBytes(self) -> bytes | str:
        """
        Downloads the file as raw bytes.

        Returns:
             bytes | str: File content in bytes, or error message.
        """
        
        blob = FireStorage.getFileInfo(self.identifierPath(), metadataOnly=False)
        if isinstance(blob, str):
            return "ERROR: Failed to retrive information from cloud storage; response: {}".format(blob)
        if blob is None:
            return "ERROR: File does not exist."
        
        return blob.download_as_bytes()
    
    @staticmethod
    def generateIDPath(store: str, filename: str):
        """
        Generates a Firebase-style path from store and filename.

        Returns:
            str: `store` and `filename` joined with `os.path.join`.
        """
        
        return f"{store}/{filename}"
    
    @staticmethod
    def from_dict(data: dict) -> 'File':
        """
        Reconstructs a File object from a dictionary.

        Args:
            data (dict): Dictionary with file fields.

        Returns:
            File: Reconstructed File instance.
        """
        
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
        return "File(name={}, store={}, id={}, contentType={}, updated={}, updateMetadata={}, forceExistence={})".format(
            self.filename,
            self.store,
            self.id if self.id != None else "None",
            self.contentType if self.contentType != None else "None",
            self.updated if self.updated != None else "None",
            self.updateMetadata if self.updateMetadata != None else "None",
            self.forceExistence if self.forceExistence != None else "None"
        )

class FileManager:
    """
    FileManager is a static interface for managing file synchronisation
    between the local filesystem and Firebase Cloud Storage.

    This class acts as a single source of file context and state management across 
    both local and cloud environments. It assumes Firebase as the source of truth
    and maintains a local context map (in-memory and JSON file-based) to represent 
    files currently recognised by the system.

    Responsibilities:
    - Ensure local folders (called "stores") exist to house files. Store names are in `FileManager.stores`
    - Maintain a JSON context file (`fmContext.json` or `FileManager.dataFile`) that maps file identifiers
      to metadata.
    - Reconcile file states during setup: download out-of-date files from Firebase,
      upload local files marked as `forceExistence`, and clean orphaned files.
    - Provide utility methods to prep, save, delete, or offload files while keeping
      the context in sync.
    - Serve as the bridge between the local system and Firebase via `FireStorage`,
      a wrapper on the Firebase Admin SDK.

    Stores are logical folders (e.g., "artefacts", "people") tracked by name, and
    each file belongs to one store. All files must be identified by a unique 
    "identifier path" of the form: `store/filename`.

    Example 1:
    ```python
    # Step 1: Setup (must be called once before usage)
    FileManager.setup()

    # Step 2: Save a local file to Firebase and register it
    FileManager.save("artefacts", "ancient_mask.jpg")

    # Step 3: Prepare a file from Firebase into the local context, i.e make it available in the system and FileManager context.
    # If the file already existed on Firebase but not locally, you can can use this make sure it's downloaded and ready for use.
    file = FileManager.prepFile("artefacts", "ancient_mask.jpg")

    # Step 4: Get signed URL, or other operations
    url = file.getSignedURL()
    ```
    
    Example 2 (Save and then offload a file):
    ```python
    # Step 1: Setup the FileManager
    FileManager.setup()

    # Step 2: Save a new file that was created in a store
    file = FileManager.save("artefacts", "fragile_scroll.jpg")

    # Step 3: Use the file (e.g., get signed URL)
    print(file.getSignedURL())

    # Step 4: Offload file locally (it remains on cloud)
    FileManager.offload(file)
    
    # Step 5: Continue doing file operations. They work because the file is still on cloud.
    print(file.getPublicURL())
    ```
    """
    
    mode = "cloud"
    dataFile = "fmContext.json"
    stores = ["FileStore", "people", "artefacts"]
    context: Dict[str, File] = {}
    initialised = False
    _dataFileLock = threading.Lock()
    
    @staticmethod
    def ensureDataFile():
        """
        Ensures that the `FileManager.dataFile` exists.

        Returns:
            bool: Always returns True after creating or verifying the file.
        """
        if not FileOps.exists(os.path.join(os.getcwd(), FileManager.dataFile), type="file"):
            with FileManager._dataFileLock:
                with open(FileManager.dataFile, "w") as f:
                    json.dump({
                        "mode": FileManager.mode
                    }, f)
        
        return True
    
    @staticmethod
    def existsInSystem(file: File) -> bool:
        """
        Checks if the file exists on the local filesystem.

        Args:
            file (File): File to check.

        Returns:
            bool: True if file exists.
        """
        
        return FileOps.exists(file.path(), type="file")
    
    @staticmethod
    def exists(file: File) -> 'tuple[bool, bool] | str':
        """
        Checks if a file exists both in Firebase and in the context.

        Args:
            file (File): File to check.

        Returns:
             Tuple[bool, bool] | str: (cloud, context) flags or error.
        """
        
        cloudFile = FireStorage.getFileInfo(file.identifierPath())
        if isinstance(cloudFile, str):
            return "ERROR: Failed to get file information from cloud storage; response: {}".format(cloudFile)
        
        cloud = cloudFile is not None
        context = False
        
        if not cloud:
            res = FileManager.removeLocally(file)
            if res != True:
                return "ERROR: Failed to remove file '{}' locally that was not found on cloud; error: {}".format(res)
            
            FileManager.debugPrint(f"'{file.identifierPath()}' was removed locally as it was not found on cloud during exists call.")
        
        # Check if file exists in context.
        if file.identifierPath() in FileManager.context:
            context = True
        
        return cloud, context
    
    @staticmethod
    def debugPrint(msg: str) -> None:
        """
        Prints a debug message if DEBUG_MODE is enabled.

        Args:
            msg (str): Message to print.
        """
        
        if os.environ.get("FM_DEBUG_MODE", os.environ.get("DEBUG_MODE", "False")) == "True":
            print("FM DEBUG: {}".format(msg))
    
    @staticmethod
    def removeLocally(file: File) -> bool | str:
        """
        Removes the file from the system and context, if present in either.

        Args:
            file (File): File to remove.

        Returns:
            bool | str: True if successful, else error message.
        """
        
        try:
            res = FileOps.deleteFile(file.path())
            if res != True:
                raise Exception(res)
            
            FileManager.debugPrint(f"'{file.identifierPath()}' removed from system.")
            
            if file.identifierPath() in FileManager.context:
                del FileManager.context[file.identifierPath()]
                FileManager.saveContext()
                
                FileManager.debugPrint(f"'{file.identifierPath()}' removed from context.")

            return True
        except Exception as e:
            return "ERROR: Failed to remove file locally; error: {}".format(e)
    
    @staticmethod
    def delete(store: str=None, filename: str=None, file: File=None) -> bool | str:
        """
        Deletes the file from Firebase and cleans up locally/contextually. Provide either `store` and `filename` or a `File` object.

        Args:
            store (str): Store the file belongs to.
            filename (str): Name of the file.
            file (File): File to delete.

        Returns:
             bool | str: True if successful, else error.
        """
        if store is None and filename is None and file is None:
            return "ERROR: Either 'store' and 'filename' or 'file' must be provided."
        
        if isinstance(file, File):
            store = file.store
            filename = file.filename
        
        if store is None or filename is None:
            return "ERROR: Could not resolve target 'store' and 'filename'."
        
        fileObject = FireStorage.getFileInfo(file.identifierPath())
        if isinstance(fileObject, str):
            return "ERROR: Failed to get information for file '{}' from cloud storage; response: {}".format(file.identifierPath(), fileObject)
        if fileObject is None:
            res = FileManager.removeLocally(file)
            if res != True:
                return "ERROR: Failed to remove file '{}' locally; error: {}".format(file.identifierPath(), res)
            
            return True
        
        res = FireStorage.deleteFile(file.identifierPath())
        if res != True:
            return "ERROR: Failed to delete file '{}' from cloud storage; response: {}".format(res)
        
        res = FileManager.removeLocally(file)
        if res != True:
            return "ERROR: Failed to remove file '{}' locally; error: {}".format(file.identifierPath(), res)
        
        return True
    
    @staticmethod
    def deleteAll() -> bool | str:
        """Deletes all files on cloud and removes them from the local filesystem and context.

        Returns:
            bool | str: True if successful, else error.
        """
        
        cloudFiles = FireStorage.listFiles(ignoreFolders=True)
        if isinstance(cloudFiles, str):
            return cloudFiles
        
        for file in cloudFiles:
            try:
                file.delete()
                FileManager.removeLocally(File(file.name.split("/")[1], file.name.split("/")[0]))
            except Exception as e:
                Logger.log("FILEMANAGER DELETEALL WARNING: Failed to delete file '{}'; error: {}".format(file.name, e))
        
        return True
    
    @staticmethod
    def cleanupNonmatchingFiles() -> bool | str:
        """
        Deletes local files from all stores that are not referenced in the current context.

        Returns:
             bool | str: True if successful, else error.
        """
        
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
    def createStores(*args: list[str]) -> bool | str:
        """
        Creates local folders for each store provided.

        Args:
            *args: Store names. If none provided, creates all known stores.

        Returns:
             bool | str: True if successful, else error.
        """
        
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
        """
        Loads the FileManager context from the data file and validates it.

        Returns:
             bool | str: True if successful, else error.
        """
        
        FileManager.ensureDataFile()
        
        # Read JSON data from data file
        data: dict | None = None
        try:
            with FileManager._dataFileLock:
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
                if not FileManager.existsInSystem(file):
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
        """
        Saves the current context to the JSON file.

        Returns:
             bool | str: True if successful, else error.
        """
        
        try:
            with FileManager._dataFileLock:
                with open(FileManager.dataFile, "w") as f:
                    data = {file.identifierPath(): file.represent() for file in FileManager.context.values()}
                    data['mode'] = FileManager.mode
                    
                    json.dump(data, f, indent=4)
        except Exception as e:
            return "ERROR: Failed to save context to file; error: {}".format(e)
        
        return True
    
    @staticmethod
    def setup() -> bool | str:
        """
        Sets up the FileManager system by synchronising local context with Firebase Cloud Storage.

        This is the **initialisation entrypoint** and must be called once before using any
        other FileManager methods like `save`, `prepFile`, or `getFromContext`.

        The setup process involves the following steps:
        1. Ensures all defined file stores (folders) exist on the local filesystem.
        2. Loads the file context from the JSON file (`FileManager.dataFile`), including:
        - Files previously tracked and valid (those existing in both the system and the context).
        - New local files that exist in the filesystem but are not in the context (auto-added).
        - Cleanup of invalid or orphaned entries.
        3. Fetches metadata from Firebase Cloud Storage for all remote files.
        4. For each file in context:
        - If `forceExistence=True` and the file is missing in Firebase, it is uploaded.
        - If file exists on cloud but is outdated locally, it is re-downloaded.
        - If marked with `updateMetadata=True`, metadata is refreshed from Firebase.
        5. Files that do not exist in Firebase and are not marked with `forceExistence`
        are deleted from both the system and context.
        6. Final context is written back to the context JSON file.

        Returns:
            bool | str: Returns True if setup was successful. Returns an error string if any
                        operation (file upload, download, sync, etc.) fails.

        Raises:
            None directly, but Firebase errors or I/O errors are caught and reported as strings.

        Example:
        ```python
        result = FileManager.setup()
        if result == True:
            print("FileManager initialised.")
        else:
            print("Error during setup:", result)
        ```
        """
        
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
                    if not FileManager.existsInSystem(FileManager.context[fileIDPath]):
                        FileManager.debugPrint(f"'{fileIDPath}': Force exist flag set but file does not exist locally. Cannot upload.")
                        continue
                    
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
                    res = FileManager.removeLocally(FileManager.context[fileIDPath])
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
                    FileManager.debugPrint(f"'{fileIDPath}': Redownloaded file due to newer version available.")
                
                # If the file was redownloaded due to update parity or if the context has an updateMetadata mandate, update the context metadata
                if FileManager.context[fileIDPath].updateMetadata or reDownloaded:
                    FileManager.debugPrint(f"'{fileIDPath}': Context data updated due to updateMetadata: {FileManager.context[fileIDPath].updateMetadata} or reDownloaded: {reDownloaded}")
                    # Update context metadata
                    FileManager.context[fileIDPath].updateData(cloudFiles[fileIDPath])
        
        FileManager.saveContext()
        FileManager.cleanupNonmatchingFiles()
        FileManager.initialised = True
        
        print("FM: FileManager setup complete. Stores: {}, Mode: {}".format(len(FileManager.stores), FileManager.mode))
        
        return True

    @staticmethod
    def prepFile(store: str=None, filename: str=None, file: File=None) -> File | str:
        """
        Prepares a file in the local context if it exists on cloud. Provide either `store` and `filename` or a `File` object.

        Args:
            store (str): Store the file belongs to.
            filename (str): Name of the file.
            file (File): File object.

        Returns:
             File | str: File if successful, else error.
        """
        if store is None and filename is None and file is None:
            return "ERROR: Either 'store' and 'filename' or 'file' must be provided."
        
        if isinstance(file, File):
            store = file.store
            filename = file.filename
        
        if store is None or filename is None:
            return "ERROR: Could not resolve target 'store' and 'filename'."
        
        if not FileManager.initialised:
            return "ERROR: FileManager has not been setup yet."
        if store not in FileManager.stores:
            return "ERROR: Store '{}' is not tracked by FileManager.".format(store)
        
        idPath = File.generateIDPath(store, filename)
        
        cloudFile = FireStorage.getFileInfo(idPath)
        if cloudFile == None:
            # File does not exist. Delete from filesystem and context if it exists            
            res = FileManager.removeLocally(File(filename, store))
            if res != True:
                Logger.log("FILEMANAGER PREPFILE WARNING: Failed to remove cloud-unmatched file '{}' locally; error: {}".format(idPath, res))
            
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
            
            # File needs to be downloaded to ensure that the file is the same as cloud, otherwise it could a same filename but different data issue
            res = FireStorage.downloadFile(newFile.path(), newFile.identifierPath())
            if res != True:
                return "ERROR: Failed to re-download file '{}' from cloud storage for data integrity; response: {}".format(newFile.identifierPath(), res)
            
            FileManager.context[newFile.identifierPath()] = newFile
            FileManager.saveContext()
            
            FileManager.debugPrint(f"'{newFile.identifierPath()}' added to context and re-downloaded for data integrity as it was found on cloud during file prep.")
        
        ## Ensure existence in the filesystem
        if not FileManager.existsInSystem(FileManager.context[idPath]):
            # File is missing and needs to be downloaded
            res = FireStorage.downloadFile(FileManager.context[idPath].path(), idPath)
            if res != True:
                return "ERROR: Failed to download missing file '{}' from cloud storage; response: {}".format(idPath, res)
            
            FileManager.debugPrint(f"'{idPath}' downloaded as it was found on cloud during file prep.")
        
        return FileManager.context[idPath]

    @staticmethod
    def save(store: str=None, filename: str=None, file: File=None) -> File | str:
        """
        Saves a file from the local system to Firebase. Provide either `store` and `filename` or a `File` object.

        Args:
            store (str): Store the file is in.
            filename (str): File name.
            file (File): File object to save.

        Returns:
             File | str: Updated File or error string.
        """
        if store is None and filename is None and file is None:
            return "ERROR: Either 'store' and 'filename' or 'file' must be provided."
        
        if isinstance(file, File):
            store = file.store
            filename = file.filename
        
        if store is None or filename is None:
            return "ERROR: Could not resolve target 'store' and 'filename'."
        
        if not FileManager.initialised:
            return "ERROR: FileManager has not been setup yet."
        if store not in FileManager.stores:
            return "ERROR: Store '{}' is not tracked by FileManager.".format(store)
        
        targetFile: File = None
        if File.generateIDPath(store, filename) in FileManager.context:
            targetFile = FileManager.context[File.generateIDPath(store, filename)]
        else:
            targetFile = File(filename, store)
        
        if not FileManager.existsInSystem(targetFile):
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
    
    @staticmethod
    def offload(file: File) -> bool | str:
        """
        Deletes a file from the local system but not Firebase.

        Args:
            file (File): File to offload.

        Returns:
             bool | str: True if successful, else error.
        """
        
        if not FileManager.initialised:
            return "ERROR: FileManager has not been setup yet."
        
        fileExists = FileManager.exists(file)
        if isinstance(fileExists, str):
            return "ERROR: Failed to check cloud and context existence of file '{}'; error: {}".format(file.identifierPath(), fileExists)
        if not fileExists[0]:
            return "ERROR: File does not exist."
        
        res = FileManager.removeLocally(file)
        if res != True:
            return "ERROR: Failed to offload file '{}' locally; error: {}".format(file.identifierPath(), res)
        
        FileManager.debugPrint(f"'{file.identifierPath()}' removed locally by offload call.")
        
        return True
    
    @staticmethod
    def getFromContext(store: str, filename: str):
        """Retrieves a file from the context (i.e `FileManager.context`).

        Args:
            store (str): The store the file is in.
            filename (str): The name of the file.

        Returns:
              File | None: The requested file or None if not found.
        """
        
        idPath = File.generateIDPath(store, filename)
        return FileManager.context.get(idPath)