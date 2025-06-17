import os, json, sys, requests, copy, threading
from typing import List, Dict, Any, Union
from services import Logger, Universal
from dotenv import load_dotenv
load_dotenv()

class Model:
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
    
    def __init__(self, name: str, filename: str, driveID: str | None=None):
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
    
    def represent(self) -> Dict[str, Any]:
        '''Represents the model as a dictionary.'''
        return {
            "name": self.name,
            "filename": self.filename,
            "driveID": self.driveID
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Model':
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
        
        return Model(
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
    
    context: Dict[str, Model] = {}
    
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
                    ModelStore.context[key] = Model.from_dict(data[key])
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
    def getModel(name: str) -> Model | None:
        '''Retrieves a model by its name from the context.'''
        return ModelStore.context.get(name)
    
    @staticmethod
    def removeModel(identifier: str | Model) -> bool:
        '''Removes a model from the context by its name or Model instance.'''
        if isinstance(identifier, Model):
            identifier = identifier.name
        
        if identifier not in ModelStore.context:
            return False
        else:
            del ModelStore.context[identifier]
        
        ModelStore.saveContext()
        return True
    
    @staticmethod
    def new(model: Model) -> bool | str:
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

class ASReport:
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
    dataFile = "archsmith.json"
    
    cache: Dict[str, ASTracer] = {}
    paused: bool = False
    
    @staticmethod
    def ensureDataFile():
        if not os.path.isfile(os.path.join(os.getcwd(), ArchSmith.dataFile)):
            with open(ArchSmith.dataFile, 'w') as f:
                json.dump({}, f, indent=4)
        
        return True
    
    @staticmethod
    def newTracer(purpose: str) -> ASTracer:
        tracer = ASTracer(purpose)
        ArchSmith.cache[tracer.id] = tracer
        
        return tracer

    @staticmethod
    def persist():
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
        ArchSmith.paused = True
        Logger.log("ARCHSMITH PAUSE: All tracers are now paused. Reports will not be tracked until resumed.")
        
        return True
    
    @staticmethod
    def resume(dropPausedReports: bool=False, autoPersist: bool=True):
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