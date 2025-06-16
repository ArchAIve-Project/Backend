import os, json, sys, requests, copy
from typing import List, Dict, Any, Union
from services import Logger
from dotenv import load_dotenv
load_dotenv()

class Model:
    def __init__(self, name: str, filename: str, driveID: str | None=None):
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
        return f"https://drive.usercontent.google.com/download?id={self.driveID}&export=download&authuser=0&confirm=t"
    
    def fileExists(self) -> bool:
        return os.path.isfile(ModelStore.modelFilePath(self.filename))
    
    def download(self) -> bool | str:
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
        return {
            "name": self.name,
            "filename": self.filename,
            "driveID": self.driveID
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Model':
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
        preppedData = {}
        for modelName in ModelStore.context:
            preppedData[modelName] = ModelStore.context[modelName].represent()
        
        with open(ModelStore.contextFilePath(), 'w') as f:
            json.dump(preppedData, f, indent=4)
        
        return True
    
    @staticmethod
    def setup():
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
        return ModelStore.context.get(name)
    
    @staticmethod
    def removeModel(identifier: str | Model) -> bool:
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