import os, json, sys, requests, copy, threading
from typing import List, Dict, Any, Union
from services import Logger, Universal
from dotenv import load_dotenv
load_dotenv()

class ModelContext:
    '''Represents a machine learning model with metadata and methods for downloading and managing the model file.
    
    Attributes:
        name (str): The name of the model.
        filename (str): The filename of the model file.
        driveID (str | None): The Google Drive ID for downloading the model file, if applicable.
        
    Methods:
        driveDownloadURL() -> str: Returns the download URL for the model file from Google Drive.
        fileExists() -> bool: Checks if the model file exists in the local filesystem.
        download() -> bool | str: Downloads the model file from Google Drive if it exists, or returns an error message.
        represent() -> Dict[str, Any]: Prepares the model's data for serialization.
        from_dict(data: Dict[str, Any]) -> 'Model': Creates a Model instance from a dictionary representation.
        __str__() -> str: Returns a string representation of the Model instance.
        
    Raises:
        ValueError: If 'name' or 'filename' is not a non-empty string.
        Exception: If the model file cannot be downloaded or if the driveID is not provided.
    
    ## Usage
    ```python
    model = Model(name="ExampleModel", filename="example_model.pth", driveID="12345abcde")
    model.download()  # Downloads the model file if it exists on Google Drive (`driveID` must be set)
    model.fileExists()  # Checks if the model file exists locally
    model.represent()  # Returns a dictionary representation of the model
    new_model = Model.from_dict(model.represent())  # Deserialize back to a Model instance
    ```
    
    Model files are stored in a directory specified by `ModelStore.rootDir`.
    
    Note: Interaction with `Model` is typically managed through the `ModelStore` class, which handles the context and storage of multiple models.
    '''
    
    def __init__(self, name: str, filename: str, driveID: str | None=None, model: Any=None, loadCallback=None):
        '''Initializes a Model instance with the given name, filename, and optional Google Drive ID.
        
        Args:
            name (str): The name of the model.
            filename (str): The filename of the model file.
            driveID (str | None): The Google Drive ID for downloading the model file, if applicable.
        Raises:
            ValueError: If 'name' or 'filename' is not a non-empty string.
        
        '''
        if not name or not isinstance(name, str):
            raise ValueError("MODEL INIT: 'name' must be a non-empty string.")
        if not filename or not isinstance(filename, str):
            raise ValueError("MODEL INIT: 'filename' must be a non-empty string.")
        
        name = name.strip()
        filename = filename.strip()
        driveID = driveID.strip() if driveID else None

        self.name = name
        self.filename = filename
        self.driveID = driveID
        self.model = model
        self.loadCallback = loadCallback
    
    def driveDownloadURL(self) -> str:
        '''Generates the Google Drive download URL for the model file.
        
        Returns:
            str: The download URL for the model file.
        
        Raises:
            Exception: If the driveID is not provided.
        '''
        
        if not self.driveID:
            raise Exception("ERROR: Cannot generate download URL; no driveID provided.")
        return f"https://drive.usercontent.google.com/download?id={self.driveID}&export=download&authuser=0&confirm=t"
    
    def fileExists(self) -> bool:
        '''Checks if the model file exists in the local filesystem.'''
        return os.path.isfile(ModelStore.modelFilePath(self.filename))
    
    def download(self) -> bool | str:
        '''Downloads the model file from Google Drive if it exists.
        
        Returns:
            bool | str: True if the download was successful, or an error message if it failed.
        
        '''
        if not self.driveID:
            return "ERROR: No driveID provided."
        
        if os.path.isfile(ModelStore.modelFilePath(self.filename)):
            Logger.log("MODEL DOWNLOAD WARNING: Model file '{}' ({}) already exists, it will be overwritten.".format(self.filename, self.name))
        
        try:
            with requests.get(self.driveDownloadURL(), stream=True) as response:
                response.raise_for_status()
                with open(ModelStore.modelFilePath(self.filename), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192): # 8 KB chunks
                        if chunk: # Filter out keep-alive chunks
                            f.write(chunk)
        except Exception as e:
            return f"ERROR: Failed to download model '{self.name}'. Error: {e}"
        
        Logger.log("MODEL DOWNLOAD: Successfully downloaded model '{}' ({}).".format(self.name, self.filename), debugPrintExplicitDeny=True)
        
        return True
    
    def load(self):
        self.model = self.loadCallback(ModelStore.modelFilePath(self.filename)) if self.loadCallback else None
        return self.model
    
    def represent(self) -> Dict[str, Any]:
        '''Represents the model as a dictionary.'''
        return {
            "name": self.name,
            "filename": self.filename,
            "driveID": self.driveID
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelContext':
        '''Creates a Model instance from a dictionary representation.
        
        Args:
            data (Dict[str, Any]): A dictionary containing the model's data.
        Returns:
            Model: A new Model instance created from the provided data.
        Raises:
            ValueError: If the data is not a dictionary or does not contain 'name' and 'filename' keys.
        
        '''
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary.")
        
        if "name" not in data or "filename" not in data:
            raise ValueError("Data must contain 'name' and 'filename' keys.")
        
        return ModelContext(
            name=data.get("name", None),
            filename=data.get("filename", None),
            driveID=data.get("driveID", None)
        )
    
    def __str__(self):
        return f"Model(name={self.name}, filename={self.filename}, driveID={self.driveID})"

class ModelStore:
    '''Manages a collection of machine learning models, providing methods to load, save, and retrieve models.
    
    Attributes:
        rootDir (str): The directory where models are stored.
        contextFile (str): The filename for the context file that stores model metadata.
        context (Dict[str, Model]): A dictionary mapping model names to Model instances.
    
    Methods:
        rootDirPath() -> str: Returns the absolute path to the root directory for models.
        contextFilePath() -> str: Returns the absolute path to the context file.
        modelFilePath(filename: str) -> str: Returns the absolute path to a specific model file.
        modelPath(name: str) -> str: Returns the absolute path to a model file based on its name.
        
        loadContext() -> bool: Loads the model context from the context file, creating it if it doesn't exist.
        saveContext() -> bool: Saves the current model context to the context file.
        
        setup() -> None: Sets up the model store by creating the root directory, loading the context, and downloading models as needed.
        getModel(name: str) -> Model | None: Retrieves a model by its name from the context.
        removeModel(identifier: str | Model) -> bool: Removes a model from the context by its name or Model instance.
        new(model: Model) -> bool | str: Adds a new model to the context, downloading it if necessary.
        registerLoadModelCallbacks(**kwargs) -> bool: Registers load callbacks for models in the context.
        loadModels(*args) -> bool: Loads models by their names, executing their load callbacks if defined. Provide no arguments to load all models.
        
    ## Usage
    ```python
    model_store = ModelStore()
    model_store.setup()  # Initializes the model store, loading existing models and downloading as needed
    model = Model(name="ExampleModel", filename="example_model.pth", driveID="12345abcde")
    model_store.new(model)  # Adds a new model to the store
    retrieved_model = model_store.getModel("ExampleModel")  # Retrieves the model by name
    model_store.removeModel("ExampleModel")  # Removes the model from the store
    ```
    
    Model files are stored in a directory specified by `ModelStore.rootDir`, and the context is saved in `ModelStore.contextFile` (typically `context.json`).
    
    Note: `ModelStore` will skip non-downloadable non-existing models during setup, and will log warnings for any issues encountered during model loading or downloading.
    '''
    
    rootDir: str = "models"
    contextFile: str = "context.json"
    
    context: Dict[str, ModelContext] = {}
    
    @staticmethod
    def rootDirPath() -> str:
        return os.path.join(os.getcwd(), ModelStore.rootDir)

    @staticmethod
    def contextFilePath() -> str:
        return os.path.join(ModelStore.rootDirPath(), ModelStore.contextFile)
    
    @staticmethod
    def modelFilePath(filename: str) -> str:
        return os.path.join(ModelStore.rootDirPath(), filename)
    
    @staticmethod
    def modelPath(name: str) -> str:
        if name in ModelStore.context:
            return ModelStore.modelFilePath(ModelStore.context[name].filename)
        else:
            raise Exception("MODELPATH ERROR: Given model name does not exist in context.")

    @staticmethod
    def loadContext() -> bool:
        '''Loads the model context from the context file, creating it if it doesn't exist.'''
        if not os.path.isfile(ModelStore.contextFilePath()):
            ModelStore.context = {}
            with open(ModelStore.contextFilePath(), 'w') as f:
                json.dump(ModelStore.context, f, indent=4)
        else:
            with open(ModelStore.contextFilePath(), 'r') as f:
                data = json.load(f)
            
            for key in data:
                try:
                    ModelStore.context[key] = ModelContext.from_dict(data[key])
                except Exception as e:
                    Logger.log(f"MODELSTORE LOADCONTEXT WARNING: Error loading model '{key}' from context (skipping); error: {e}")
                    if key in ModelStore.context:
                        del ModelStore.context[key]

        return True
    
    @staticmethod
    def saveContext() -> bool:
        '''Saves the current model context to the context file.'''
        preppedData = {}
        for modelName in ModelStore.context:
            preppedData[modelName] = ModelStore.context[modelName].represent()
        
        with open(ModelStore.contextFilePath(), 'w') as f:
            json.dump(preppedData, f, indent=4)
        
        return True
    
    @staticmethod
    def setup():
        '''Sets up the model store by creating the root directory, loading the context, and downloading models as needed.'''
        if not os.path.isdir(ModelStore.rootDirPath()):
            os.makedirs(ModelStore.rootDirPath())
        
        ModelStore.loadContext()
        
        for model in [x for x in ModelStore.context.values()]:
            if not model.fileExists():
                if model.driveID:
                    try:
                        print("MODELSTORE SETUP: Downloading model '{}'...".format(model.name))
                        res = model.download()
                        if res != True:
                            raise Exception(res)
                    except Exception as e:
                        print("MODELSTORE SETUP WARNING: Failed to download model '{}'; model will be excluded from context. Error: {}".format(model.name, e))
                        del ModelStore.context[model.name]
                else:
                    print("MODELSTORE SETUP WARNING: Model file not found for non-downloadable model '{}'. Model will be excluded from context.".format(model.name))
                    del ModelStore.context[model.name]
            # else:
            #     print("MODELSTORE: '{}' loaded successfully.".format(model.name))
        
        ModelStore.saveContext()
        
        print("MODELSTORE: Setup complete. {} model(s) loaded.".format(len(ModelStore.context)))
    
    @staticmethod
    def getModel(name: str) -> ModelContext | None:
        '''Retrieves a model by its name from the context.'''
        return ModelStore.context.get(name)
    
    @staticmethod
    def removeModel(identifier: str | ModelContext) -> bool:
        '''Removes a model from the context by its name or Model instance.'''
        if isinstance(identifier, ModelContext):
            identifier = identifier.name
        
        if identifier not in ModelStore.context:
            return False
        else:
            del ModelStore.context[identifier]
        
        ModelStore.saveContext()
        return True
    
    @staticmethod
    def new(model: ModelContext) -> bool | str:
        '''Adds a new model to the context, downloading it if necessary. Ensure that model file already exists or can be downloaded with `driveID`.'''
        if not model.fileExists():
            if model.driveID:
                try:
                    print("MODELSTORE NEW: Downloading model '{}'...".format(model.name))
                    res = model.download()
                    if res != True:
                        raise Exception(res)
                except Exception as e:
                    Logger.log("MODELSTORE NEW ERROR: Error in downloading model '{}'; error: {}".format(model.name, e))
                    return "ERROR: Failed to download model '{}'; it will not be added. Error: {}".format(model.name, e)
            else:
                Logger.log("MODELSTORE NEW ERROR: Model file not found for non-downloadable model '{}'. It will not be added.".format(model.name))
                return "ERROR: Model file not found for non-downloadable model '{}'. It will not be added.".format(model.name)
        
        ModelStore.context[model.name] = model
        ModelStore.saveContext()
        
        Logger.log("MODELSTORE NEW: Model '{}' added successfully.".format(model.name))
        
        return True

    @staticmethod
    def registerLoadModelCallbacks(**kwargs):
        for name, callback in kwargs.items():
            if name in ModelStore.context:
                ModelStore.context[name].loadCallback = callback
            else:
                print("MODELSTORE REGISTERLOADMODELCALLBACK ERROR: Model '{}' does not exist in context.".format(name))
        
        return True
    
    @staticmethod
    def loadModels(*args):
        if len(args) == 0:
            for name in ModelStore.context:
                if ModelStore.context[name].loadCallback:
                    ModelStore.context[name].load()
            print("MODELSTORE LOADMODELS: All models loaded successfully.")
        else:
            for name in args:
                if name in ModelStore.context:
                    if ModelStore.context[name].loadCallback:
                        ModelStore.context[name].load()
                    else:
                        raise Exception("MODELSTORE LOADMODELS ERROR: Model '{}' does not have a load callback defined.".format(name))
                else:
                    raise Exception("MODELSTORE LOADMODELS ERROR: Model '{}' does not exist in context.".format(name))
            print("MODELSTORE LOADMODELS: Models loaded successfully: {}".format(", ".join(args)))
        
        return True

class ASReport:
    """
    ASReport represents a structured report containing information about a specific event or message, 
    including its source, message content, optional extra data, creation timestamp, and thread information.
    
    ### Args
        source (str): The origin or source of the report.
        message (str): The main message or description of the report.
        extraData (dict | None): Optional additional data relevant to the report.
    
    ### Attributes
        source (str): The origin or source of the report.
        message (str): The main message or description of the report.
        extraData (dict): Optional additional data relevant to the report.
        created (str): The UTC timestamp string when the report was created.
        threadInfo (str): Information about the thread where the report was created.
    
    ### Methods
        represent() -> Dict[str, Any]:
            Returns a dictionary representation of the report instance.
        from_dict(data: Dict[str, Any]) -> 'ASReport':
            Creates an ASReport instance from a dictionary.
        threadInfoString() -> str:
            Returns a string containing the current thread's identifier, name, and whether it is the main thread.
    """
    def __init__(self, source: str, message: str, extraData: dict | None=None):
        self.source = source
        self.message = message
        self.extraData = extraData if extraData else {}
        self.created = Universal.utcNowString()
        self.threadInfo = ASReport.threadInfoString()
    
    def represent(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "message": self.message,
            "extraData": self.extraData,
            "created": self.created,
            "threadInfo": self.threadInfo
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ASReport':
        return ASReport(
            source=data.get("source"),
            message=data.get("message"),
            extraData=data.get("extraData"),
            created=data.get("created"),
            threadInfo=data.get("threadInfo")
        )
    
    @staticmethod
    def threadInfoString() -> str:
        return f"{threading.get_ident()} {threading.current_thread().name} {isinstance(threading.current_thread(), threading._MainThread)}"

class ASTracer:
    """
    ASTracer is a class for tracing and managing analysis session reports within the ArchAIve Backend system.
    
    ### Args
        purpose (str): A description or purpose for the tracer instance.
    
    ### Attributes
        id (str): Unique identifier for the tracer instance.
        purpose (str): Description or purpose of the tracer.
        created (str): UTC timestamp string when the tracer was created.
        reports (List[ASReport]): List of active reports associated with the tracer.
        pausedReports (List[ASReport]): List of reports collected while the system is paused.
        threadInfo (str): Information about the thread in which the tracer was created.
        started (Optional[str]): UTC timestamp string when the tracer was started.
        finished (Optional[str]): UTC timestamp string when the tracer was finished.
    
    ### Methods
        __init__(purpose: str):
            Initializes a new ASTracer instance with the given purpose.
        start():
            Marks the tracer as started by recording the current UTC time.
        end():
            Marks the tracer as finished by recording the current UTC time.
        addReport(report: ASReport) -> bool:
            Adds a report to the tracer. If the system is paused, the report is added to pausedReports.
            Returns True if the report was added successfully, False otherwise.
        explicitResume() -> int:
            Moves all paused reports to the main reports list and clears pausedReports.
            Returns the number of reports restored.
        threadInfoString() -> str:
            Static method that returns a string with the current thread's identifier, name, and whether it is the main thread.
        represent() -> Dict[str, Any]:
            Returns a dictionary representation of the tracer and its reports.
        from_dict(data: Dict[str, Any]) -> 'ASTracer':
            Static method that creates an ASTracer instance from a dictionary representation.
    """
    def __init__(self, purpose: str):
        self.id = Universal.generateUniqueID()
        self.purpose = purpose
        self.created = Universal.utcNowString()
        self.reports: List[ASReport] = []
        self.pausedReports: List[ASReport] = []
        self.threadInfo = ASTracer.threadInfoString()
        self.started = None
        self.finished = None
    
    def start(self):
        self.started = Universal.utcNowString()
    
    def end(self):
        self.finished = Universal.utcNowString()
    
    def addReport(self, report: ASReport) -> bool:
        '''Adds a report to the tracer.'''
        if not self.started:
            self.start()
        if self.finished:
            Logger.log("ASTRACER ADDREPORT ERROR: Cannot add report to finished tracer (ID: {}).".format(self.id))
            return False
        
        if not isinstance(report, ASReport):
            Logger.log("ASTRACER ADDREPORT ERROR: Report must be an instance of ASReport. (ID: {})".format(self.id))
            return False
        
        if ArchSmith.paused:
            Logger.log("ASTRACER ADDREPORT WARNING: ArchSmith system paused; saving to paused reports. (ID: {}, Report: {})".format(self.id, report.source))
            self.pausedReports.append(report)
        else:
            self.reports.append(report)
        
        return True
    
    def explicitResume(self) -> int:
        '''Resumes the tracer, by adding all paused reports to the main report list.'''
        for report in self.pausedReports:
            self.reports.append(report)
        
        restoredCount = len(self.pausedReports)
        self.pausedReports.clear()
        
        Logger.log("ASTRACER EXPLICITRESUME: Resumed tracer (ID: {}) with {} restored reports.".format(self.id, restoredCount))
        return restoredCount
    
    @staticmethod
    def threadInfoString() -> str:
        return f"{threading.get_ident()} {threading.current_thread().name} {isinstance(threading.current_thread(), threading._MainThread)}"
    
    def represent(self) -> Dict[str, Any]:
        '''Represents the tracer as a dictionary.'''
        return {
            "id": self.id,
            "purpose": self.purpose,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "reports": [report.represent() for report in self.reports],
            "threadInfo": self.threadInfo
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ASTracer':
        tracer = ASTracer(
            purpose=data.get("purpose", "")
        )
        tracer.id = data.get("id")
        tracer.created = data.get("created")
        tracer.started = data.get("started")
        tracer.finished = data.get("finished")
        tracer.reports = [ASReport.from_dict(report) for report in data.get("reports", [])]
        tracer.threadInfo = data.get("threadInfo")
        return tracer

class ArchSmith:
    """
    ArchSmith is a static utility class for managing ASTracer instances and persisting their state to a JSON file.
    It provides tracer creation, data file management, and controls for pausing, resuming, and shutting down tracer operations.
    
    Attributes:
        dataFile (str): The filename for storing tracer data in JSON format.
        cache (Dict[str, ASTracer]): A cache mapping tracer IDs to ASTracer instances.
        paused (bool): Indicates whether all tracers are currently paused.
    
    Static Methods:
        checkPermissions():
            Checks if ArchSmith is enabled via the "ARCHSMITH_ENABLED" environment variable.
        setup():
            Initializes ArchSmith by checking permissions and ensuring the data file exists.
        ensureDataFile():
            Ensures the data file exists; creates it if missing.
        newTracer(purpose: str) -> ASTracer:
            Creates a new ASTracer for the given purpose and adds it to the cache.
        persist():
            Persists the current state of all tracers to the data file and removes finished tracers from the cache.
        pause():
            Pauses all tracers and prevents further report tracking until resumed.
        resume(dropPausedReports: bool=False, autoPersist: bool=True):
            Resumes all tracers, optionally dropping paused reports or restoring them, and optionally persists state.
        shutdown(gracefully: bool=True):
            Shuts down ArchSmith, optionally persisting all tracer data before clearing the cache and resetting state.
    
    ### Usage Guide
    - ArchSmith has been designed to be a centralized utility for tracer management and report processing.
    - The utility taps on a tracer's memory reference, which is internally stored, allowing for report collection from any thread, process or part of system execution.
    - It is highly recommended to call `ArchSmith.persist()` periodically to ensure that all reports are saved to disk.
    - For efficiency, ArchSmith will only keep tracers in memory that are actively being used, and will automatically remove finished tracers from the cache.
    - Use `ArchSmith.pause()` to temporarily halt report collection, and `ArchSmith.resume()` to continue.
    
    Sample code:
    ```python
    import time
    from addons import ArchSmith, ASTracer, ASReport
    from services import ThreadManager
    
    ThreadManager.initDefault()
    if ArchSmith.setup() != True:  # Ensure ArchSmith is set up and ready to use
        exit()
    
    def asyncWork(tracer: ASTracer):
        print("... Some work here ...")
        tracer.addReport(ASReport(source="ASYNCWORK", message="Work was done successfully.", extraData={"result": "success"}))
        tracer.end()
    
    tracer = ArchSmith.newTracer("Example purpose")
    ThreadManager.defaultProcessor.addJob(asyncWork, tracer) # Add a job to execute asyncWork on a background thread immediately
    
    time.sleep(2) # Simulate some time passing
    ArchSmith.persist()  # Persist the tracer data to the JSON file. This should also remove the tracer from the ArchSmith.cache, as tracer.end() was called.
    ```
    
    Note: Though ArchSmith is designed to be thread-safe, the user should ensure that the `ArchSmith.persist()` method is called in a controlled manner, or at least in a singular context.
    """
    dataFile = "archsmith.json"
    
    cache: Dict[str, ASTracer] = {}
    paused: bool = False
    
    @staticmethod
    def checkPermissions():
        return os.environ.get("ARCHSMITH_ENABLED", "False") == "True"
    
    @staticmethod
    def setup():
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        ArchSmith.ensureDataFile()
        return True
    
    @staticmethod
    def ensureDataFile():
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        if not os.path.isfile(os.path.join(os.getcwd(), ArchSmith.dataFile)):
            with open(ArchSmith.dataFile, 'w') as f:
                json.dump({}, f, indent=4)
        
        return True
    
    @staticmethod
    def newTracer(purpose: str) -> ASTracer:
        """
        Creates a new ASTracer instance for the specified purpose if permissions are granted.
        Args:
            purpose (str): The purpose or description for the new tracer.
        Returns:
            ASTracer: The newly created tracer object if permissions are granted.
            str: An error message if ArchSmith does not have permission to operate.
        """
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        tracer = ASTracer(purpose)
        ArchSmith.cache[tracer.id] = tracer
        
        return tracer

    @staticmethod
    def persist():
        """
        Persists the current state of ArchSmith's cache to a data file.
        This function performs the following steps:
        1. Checks if ArchSmith has the necessary permissions to operate. Returns an error message if not.
        2. Ensures the data file exists.
        3. Creates a deep copy of the current cache.
        4. Loads existing data from the data file.
        5. Updates the data with the representations of tracers from the copied cache.
        6. Removes finished tracers from the original cache.
        7. Writes the updated data back to the data file.
        Returns:
            str: An error message if permissions are insufficient, otherwise None.
        """
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        ArchSmith.ensureDataFile()
        copiedCache = copy.deepcopy(ArchSmith.cache)
        
        with open(ArchSmith.dataFile, "r") as f:
            data: dict = json.load(f)
        
        data.update({tracer.id: tracer.represent() for tracer in copiedCache.values()})
        
        for tracerID in list(copiedCache.keys()):
            if copiedCache[tracerID].finished:
                del ArchSmith.cache[tracerID]
        
        with open(ArchSmith.dataFile, "w") as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def pause():
        """
        Pauses all tracers in ArchSmith, preventing reports from being tracked until resumed.
        Returns:
            bool or str: Returns True if the operation is successful. Returns an error message string if ArchSmith lacks the necessary permissions.
        """
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        ArchSmith.paused = True
        Logger.log("ARCHSMITH PAUSE: All tracers are now paused. Reports will not be tracked until resumed.")
        
        return True
    
    @staticmethod
    def resume(dropPausedReports: bool=False, autoPersist: bool=True):
        """
        Resumes the ArchSmith operation, optionally dropping or restoring paused reports and persisting state.
        Args:
            dropPausedReports (bool, optional): If True, all paused reports are dropped and cleared from tracers. 
                If False, attempts to restore all paused reports. Defaults to False.
            autoPersist (bool, optional): If True, persists the current state after resuming. Defaults to True.
        Returns:
            bool or str: Returns True if the operation was successful, or an error message string if permissions are insufficient.
        """
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        ArchSmith.paused = False
        
        if dropPausedReports:
            dropCount = 0
            for tracer in ArchSmith.cache.values():
                dropCount += len(tracer.pausedReports)
                tracer.pausedReports.clear()
            Logger.log("ARCHSMITH RESUME: Dropped {} all paused reports.".format(dropCount))
        else:
            restoreCount = 0
            for tracer in ArchSmith.cache.values():
                restoreCount += tracer.explicitResume()
            Logger.log("ARCHSMITH RESUME: Resumed all tracers with {} paused reports restored.".format(restoreCount))

        if autoPersist:
            ArchSmith.persist()
        
        return True
    
    @staticmethod
    def shutdown(gracefully: bool=True):
        """
        Shuts down the ArchSmith system, either gracefully or forcefully.
        Args:
            gracefully (bool, optional): If True, persists all tracers and reports before shutdown. 
                If False, shuts down immediately without persisting. Defaults to True.
        Returns:
            bool or str: Returns True if shutdown is successful, or an error message string if permissions are insufficient.
        Behavior:
            - Checks if ArchSmith has permission to operate.
            - If 'gracefully' is True, ends all tracers, persists data, and logs the shutdown.
            - If 'gracefully' is False, logs a force shutdown without persisting data.
            - Clears the ArchSmith cache and resets the paused state.
        """
        if not ArchSmith.checkPermissions():
            return "ERROR: ArchSmith does not have permission to operate."
        
        if gracefully:
            for tracer in ArchSmith.cache.values():
                tracer.end()
            ArchSmith.persist()
            Logger.log("ARCHSMITH SHUTDOWN: Persisted all tracers and reports before shutdown.")
        else:
            Logger.log("ARCHSMITH SHUTDOWN: Force shutdown without persisting tracers and reports.")
        
        ArchSmith.cache.clear()
        ArchSmith.paused = False
        
        return True