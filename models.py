import torch
from enum import Enum
from database import DI, DIRepresentable, DIError
from fm import File, FileManager
from utils import Ref
from services import Universal, Logger, FileOps
from typing import List, Dict, Any, Literal
from faceRecog import FaceEmbedding

class Artefact(DIRepresentable):
    """
    Represents an artefact (e.g., image or document) within the DI system.

    Artefacts contain metadata and are associated with stored files. This class handles the creation,
    persistence, file management, and metadata binding of artefact objects.
    """
    def __init__(self, name: str, image: str, metadata: 'Metadata | Dict[str, Any] | None', public: bool = False, created: str=None, metadataLoaded: bool=True, id: str = None):
        """
        Initializes a new Artefact instance.

        Args:
            name: Name of the artefact.
            image: Filename of the artefact image.
            metadata: Metadata object or raw metadata dictionary.
            public: Whether the artefact is publicly accessible. Defaults to False.
            created: UTC timestamp string of creation. Auto-generated if None.
            metadataLoaded: Whether to load metadata during initialization. Defaults to True.
            id: Unique identifier. Auto-generated if None.
        """
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if metadataLoaded:
            if isinstance(metadata, dict):
                metadata = Metadata.fromMetagen(id, metadata)
            if not isinstance(metadata, Metadata):
                metadata = Metadata(artefactID=id)
        else:
            metadata = None

        self.id = id
        self.name = name
        self.image = image
        self.public = public
        self.created = created
        self.metadata: Metadata | None = metadata
        self.metadataLoaded: bool = metadataLoaded
        self.originRef = Artefact.ref(self.id)
    
    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)
    
    def reload(self):
        """
        Reloads the artefact state from the data store.
        
        Returns:
            bool: True if reload succeeded

        Raises:
            Exception: If data loading fails
        """
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("ARTEFACT RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("ARTEFACT RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("ARTEFACT RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True

    def getFMFile(self) -> File:
        """
        Constructs and returns a FileManager-compatible File object from the artefact's image name information.
        
        Returns:
            File: A File object representing the artefact's image path.
        """
        return File(self.image, "artefacts")
    
    def fmSave(self):
        """
        Saves the artefact's image file via FileManager.

        Returns:
            File | str: The updated File object if the save succeeded, or an error string otherwise.

        Notes:
            This method constructs a FileManager-compatible `File` object using the artefact's image name.
        """
        fmFile = self.getFMFile()
        return FileManager.save(fmFile.store, fmFile.filename)
    
    def fmDelete(self):
        """
        Deletes the artefact's image file via FileManager.

        Returns:
            bool | str: True if deletion succeeded, or an error message string if it failed.

        Notes:
            Constructs a FileManager-compatible `File` object using the artefact's image name.
        """
        fmFile = self.getFMFile()
        return FileManager.delete(file=fmFile)

    def represent(self) -> Dict[str, Any]:
        return {
            'public': self.public,
            'name': self.name,
            'image': self.image,
            'created': self.created,
            'metadata': self.metadata.represent() if self.metadata is not None else None
        }
    
    def __str__(self):
        return "Artefact(id='{}', name='{}', image='{}', public={}, created='{}', metadataLoaded={}, metadata={})".format(
            self.id,
            self.name,
            self.image,
            self.public,
            self.created,
            self.metadataLoaded,
            self.metadata.represent() if isinstance(self.metadata, Metadata) else "None"
        )

    @staticmethod
    def rawLoad(data: dict, artefactID: str | None=None, includeMetadata: bool=True) -> 'Artefact':
        """
        Constructs an Artefact from raw data dictionary.

        Args:
            data (dict): Raw artefact data. 
            artefactID (str, optional): Optional ID for the artefact. Will be auto-generated if not provided.
            includeMetadata (bool, optional): Whether to load metadata. Defaults to True.

        Returns:
            Artefact: Constructed artefact object.
        """
        requiredParams = ['public', 'name', 'image', 'created', 'metadata']
        for param in requiredParams:
            if param not in data:
                if param == 'public':
                    data['public'] = False
                else:
                    data[param] = None

        metadata = None
        if includeMetadata and isinstance(data['metadata'], dict) and isinstance(artefactID, str):
            metadata = Metadata.rawLoad(artefactID, data['metadata'])

        output = Artefact(
            name=data['name'], 
            image=data['image'],
            metadata=metadata,
            public=data['public'],
            created=data['created'],
            metadataLoaded=includeMetadata,
            id=artefactID
        )

        return output

    @staticmethod
    def load(id: str = None, name: str=None, image: str=None, includeMetadata: bool=True) -> 'List[Artefact] | Artefact | None':
        """
        Loads artefacts from the DI system.

        Args:
            id (str): Specific artefact ID.
            name (str): Artefact name to filter by.
            image (str): Artefact image name to filter by.
            includeMetadata (bool): Whether to include metadata.

        Returns:
            Artefact | List[Artefact] | Single artefact, list, or None.
        """
        if id is not None:
            data = DI.load(Artefact.ref(id))
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if data is None:
                if name is not None or image is not None:
                    return Artefact.load(name=name, image=image, includeMetadata=includeMetadata)
                else:
                    return None
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return Artefact.rawLoad(data, id, includeMetadata)
        else:
            data = DI.load(Ref("artefacts"))
            if data == None:
                return []
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Failed to load dictionary artefacts data; response: {}".format(data))

            artefacts: Dict[str, Artefact] = {}
            for id in data:
                if isinstance(data[id], dict):
                    artefacts[id] = Artefact.rawLoad(data[id], id, includeMetadata)
            
            if name is None and image is None:
                return list(artefacts.values())
            
            for artefactID in artefacts:
                targetArtefact = artefacts[artefactID]
                if name != None and targetArtefact.name == name:
                    return targetArtefact
                elif image != None and targetArtefact.image == image:
                    return targetArtefact
            
            return None
    
    @staticmethod
    def fromFMFile(file: File) -> 'Artefact | None':
        return Artefact(
            name=file.filenameWithoutExtension(),
            image=file.filename,
            metadata=None
        )

    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("artefacts", id)

class Metadata(DIRepresentable):
    def __init__(self, artefactID: str=None, dataObject: 'MMData | HFData'=None):
        """
        Wrapper for artefact metadata, either MMData or HFData, with helper methods for storage and format conversion.
        Initializes a Metadata wrapper with MMData or HFData.

        Args:
            artefactID (str): Artefact ID to associate.
            dataObject (MMData | HFData): Metadata content.
        
        Raises:
            Exception: If dataObject is not MMData or HFData.
        """
        self.raw: MMData | HFData | None = None

        if dataObject is not None:
            if not (isinstance(dataObject, MMData) or isinstance(dataObject, HFData)):
                raise Exception("METADATA INIT ERROR: Expected data type 'MMData | HFData' for parameter 'dataObject'.")

            self.raw = dataObject
        
        self.originRef = Metadata.ref(artefactID)
        
    def save(self):
        return self.raw.save() if self.raw is not None else False
    
    def reload(self):
        return self.raw.reload() if self.raw is not None else False
    
    def destroy(self):
        return self.raw.destroy() if self.raw is not None else False
    
    def represent(self):
        return self.raw.represent() if self.raw is not None else None
    
    def isMM(self) -> bool:
        return isinstance(self.raw, MMData)
    
    def isHF(self) -> bool:
        return isinstance(self.raw, HFData)
    
    @staticmethod
    def rawLoad(artefactID: str, data: Dict[str, Any]) -> 'Metadata':
        """
        WARNING:
        Do NOT use this method to generate a Metadata object from `MetadataGenerator.generate` output.
        Use `Metadata.fromMetagen` instead.

        Determines and loads the correct metadata type from raw dictionary.

        Args:
            artefactID (str): Artefact ID.
            data (dict): Raw metadata.

        Returns:
            Metadata: Loaded metadata wrapper.

        Raises:
        Exception: If metadata type cannot be determined from the input dictionary.
        """
        if 'tradCN' in data:
            return Metadata(artefactID, MMData.rawLoad(data, artefactID))
        elif 'faceFiles' in data:
            return Metadata(artefactID, HFData.rawLoad(data, artefactID))
        else:
            raise Exception("METADATA RAWLOAD ERROR: Unable to determine metadata type from raw dictionary data.")

    @staticmethod
    def load(artefactID: str) -> 'Metadata | None':
        data = DI.load(Metadata.ref(artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("METADATA LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("METADATA LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        return Metadata.rawLoad(artefactID, data)

    @staticmethod
    def fromMetagen(artefactID: str, metagenDict: dict) -> 'Metadata':
        """
        Constructs Metadata from `metadataGenerator.generate` dictionary (from form(s) or user input).

        Args:
            artefactID (str): Associated artefact ID.
            metagenDict (dict): Raw metadata dictionary from user input.

        Returns:
            Metadata: Constructed Metadata object.

        Raises:
            Exception: If metadata type cannot be determined from input.
        """
        if 'traditional_chinese' in metagenDict:
            mm = MMData(
                artefactID=artefactID,
                tradCN=metagenDict.get('traditional_chinese'),
                preCorrectionAcc=metagenDict.get('pre_correction_accuracy'),
                postCorrectionAcc=metagenDict.get('post_correction_accuracy'),
                simplifiedCN=metagenDict.get('simplified_chinese'),
                english=metagenDict.get('english_translation'),
                summary=metagenDict.get('summary'),
                nerLabels=metagenDict.get('ner_labels'),
                corrected=metagenDict.get('correction_applied', False)
            )
            return Metadata(artefactID, mm)
        elif 'faceFiles' in metagenDict:
            hf = HFData(
                artefactID=artefactID,
                faceFiles=metagenDict.get('faceFiles'),
                caption=metagenDict.get('caption')
            )
            return Metadata(artefactID, hf)
        else:
            raise Exception("METADATA FROMMETAGEN ERROR: Unable to determine metadata type from metagen dictionary data.")

    @staticmethod
    def ref(artefactID: str) -> Ref:
        return Ref("artefacts", artefactID, "metadata")


class MMData(DIRepresentable):
    def __init__(self, artefactID: str, tradCN: str, preCorrectionAcc: str, postCorrectionAcc: str, simplifiedCN: str, english: str, summary: str, nerLabels: str, corrected: bool = False):
        """
        Metadata for meeting minutes artefacts.
    
        Contains structured data from meeting minutes including Chinese text (both traditional
        and simplified), English translations, accuracy metrics for text processing,
        summaries, and named entity labels. Tracks correction status and processing history.

        Args:
            artefactID (str): Associated artefact identifier.
            tradCN (str): Traditional Chinese text from meeting minutes.
            corrected (bool): Flag indicating if text has been manually corrected.
            preCorrectionAcc (str): Accuracy score before any corrections.
            postCorrectionAcc (str): Accuracy score after corrections.
            simplifiedCN (str): Simplified Chinese version of the text.
            english (str): English translation of the meeting minutes.
            summary (str): Concise summary of the meeting content.
            nerLabels (str): Named entity recognition labels extracted from text.
        """
        self.artefactID = artefactID
        self.tradCN = tradCN
        self.corrected = corrected
        self.preCorrectionAcc = preCorrectionAcc
        self.postCorrectionAcc = postCorrectionAcc
        self.simplifiedCN = simplifiedCN
        self.english = english
        self.summary = summary
        self.nerLabels = nerLabels
        self.originRef = MMData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "tradCN": self.tradCN,
            "corrected": self.corrected,
            "preCorrectionAcc": self.preCorrectionAcc,
            "postCorrectionAcc": self.postCorrectionAcc,
            "simplifiedCN": self.simplifiedCN,
            "english": self.english,
            "summary": self.summary,
            "nerLabels": self.nerLabels
        }

    @staticmethod
    def rawLoad(data: Dict[str, Any], artefactID: str) -> 'MMData':
        requiredParams = ['tradCN', 'corrected', 'preCorrectionAcc', 'postCorrectionAcc', 'simplifiedCN', 'english', 'summary', 'nerLabels']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'corrected':
                    data['corrected'] = False
                else:
                    data[reqParam] = None

        if not isinstance(data['corrected'], bool):
            data['corrected'] = False

        return MMData(
            artefactID=artefactID,
            tradCN=data['tradCN'],
            corrected=data['corrected'],
            preCorrectionAcc=data['preCorrectionAcc'],
            postCorrectionAcc=data['postCorrectionAcc'],
            simplifiedCN=data['simplifiedCN'],
            english=data['english'],
            summary=data['summary'],
            nerLabels=data['nerLabels']
        )
        
    @staticmethod
    def load(artefactID: str) -> 'MMData | None':
        data = DI.load(MMData.ref(artefactID))
        if data == None:
            return None
        if isinstance(data, DIError):
            raise Exception("MMDATA LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("MMDATA LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        return MMData.rawLoad(data, artefactID)

    @staticmethod
    def ref(artefactID: str) -> Ref:
        return Ref("artefacts", artefactID, "metadata")

class HFData(DIRepresentable):
    def __init__(self, artefactID: str, faceFiles: list[str], caption: str, addInfo: str=None):
        """
        Metadata for human figure(s) in artefacts.
    
        Contains information about human figures detected in images, including references
        to face image files, captions describing the scene (e.g., "people in a meeting"), and optional additional
        information. Used for artefacts containing human imagery that requires annotation.

        Args:
            artefactID (str): Associated artefact identifier.
            faceFiles (list[str]): List of image files containing detected faces.
            caption (str): Descriptive caption for the human figures.
            addInfo (str | None): Additional context about the figures if available.
        """
        self.artefactID = artefactID
        self.faceFiles = faceFiles
        self.caption = caption
        self.addInfo = addInfo
        self.originRef = HFData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "faceFiles": self.faceFiles,
            "caption": self.caption,
            "addInfo": self.addInfo
        }

    @staticmethod
    def rawLoad(data: Dict[str, Any], artefactID: str) -> 'HFData':
        requiredParams = ['faceFiles', 'caption', 'addInfo']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'faceFiles':
                    data['faceFiles'] = []
                else:
                    data[reqParam] = None

        return HFData(
            artefactID=artefactID,
            faceFiles=data.get('faceFiles', []),
            caption=data.get('caption'),
            addInfo=data.get('addInfo', None)
        )

    @staticmethod
    def load(artefactID: str) -> 'HFData | None':
        data = DI.load(HFData.ref(artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("HFDATA LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("HFDATA LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        return HFData.rawLoad(data, artefactID)

    @staticmethod
    def ref(artefactID: str) -> Ref:
        return Ref("artefacts", artefactID, "metadata")
    
class Book(DIRepresentable):
    def __init__(self, title: str, subtitle: str, mmIDs: List[str], id: str=None):
        """
        Represents a curated collection of MMData artefacts as a Book.

        Args:
            title (str): The title of the book.
            subtitle (str): Subtitle or additional description.
            mmIDs (List[str]): A list of artefact IDs that are part of the book.
            id (str, optional): Unique ID for the book. Auto-generated if not provided.
        """
        if id is None:
            id = Universal.generateUniqueID()
        
        self.id = id
        self.title = title
        self.subtitle = subtitle
        self.mmIDs = mmIDs
        self.originRef = Book.ref(self.id)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "mmIDs": self.mmIDs
        }
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BOOK RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BOOK RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BOOK RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    @staticmethod
    def rawLoad(data: dict, bookID: str | None=None) -> 'Book':
        reqParams = ['title', 'subtitle', 'mmIDs']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'mmIDs':
                    data[reqParam] = []
                else:
                    data[reqParam] = None
        
        return Book(
            title=data.get('title'),
            subtitle=data.get('subtitle'),
            mmIDs=data.get('mmIDs', []),
            id=bookID
        )
    
    @staticmethod
    def load(id: str=None, title: str=None, withMMID: str=None) -> 'Book | List[Book] | None':
        """
        Loads a book(s) by ID, title, or MMData artefact ID.

        Args:
            id (str): Book ID.
            title (str): Book title to search for.
            withMMID (str, optional): MMData artefact ID to search in book's mmIDs.

        Returns:
            Book | List[Book] | None
        """
        if id != None:
            data = DI.load(Book.ref(id))
            if data is None:
                if title is not None or withMMID is not None:
                    return Book.load(title=title, withMMID=withMMID)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("BOOK LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BOOK LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Book.rawLoad(data, id)

        else:
            data = DI.load(Ref("books"))
            if data is None:
                return None
            if isinstance(data, DIError):
                raise Exception("BOOK LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BOOK LOAD ERROR: Failed to load dictionary books data; response: {}".format(data))

            books: Dict[str, Book] = {}
            for id in data:
                if isinstance(data[id], dict):
                    books[id] = Book.rawLoad(data[id], id)
            
            if title is None and withMMID is None:
                return list(books.values())
            
            for book in books.values():
                if title is not None and book.title == title:
                    return book
                elif withMMID is not None and withMMID in book.mmIDs:
                    return book

            return None

    @staticmethod
    def ref(id: str):
        return Ref("books", id)

class Batch(DIRepresentable):
    """A collection of references to artefacts that are processed together. A `DIRepresentable` model.
    
    Notes:
    - `Batch` is a larger model that has a collection of `BatchArtefact` objects, which are references to `Artefact` objects.
    - Processing job data associated with a batch is stored as `BatchProcessingJob` under the `job` attribute.
    - The memory size of this model can get quite extensive if all sub-units of data are loaded and there are many artefacts.
    - Each `BatchArtefact` has a `stage` attribute that indicates its processing stage, among other parameters to store processing information.
    - Each `job: BatchProcessingJob` has a `status` attribute that indicates the status of the processing job. It also contains a `jobID`, which is meant to be the `ThreadManager` job ID.
    - Methods below are excluding those already defined in `DIRepresentable`.
    
    Attributes:
        id (str): Unique identifier for the batch.
        userID (str): ID of the user who created the batch.
        artefacts (Dict[str, BatchArtefact]): Dictionary of `BatchArtefact` objects keyed by artefact ID.
        job (BatchProcessingJob | None): The processing job associated with the batch, if any.
        user (User | None): The user object associated with the batch, if loaded.
        created (str): Timestamp of when the batch was created.
        originRef (Ref): Reference to the batch in the database.
    
    Methods:
        `save(checkIntegrity: bool=True)`: Saves the batch to the database, checking integrity if specified.
        `represent()`: Returns a dictionary representation of the batch.
        `reload()`: Reloads the batch data from the database.
        `getUser()`: Loads the user associated with the batch.
        `loadAllArtefacts(withMetadata: bool=False)`: Loads all artefacts in the batch, optionally including metadata. Artefacts are loaded internally in the `BatchArtefact` objects (`artefact` attribute).
        `get(artefact: Artefact | str)`: Retrieves a `BatchArtefact` by its artefact ID or `Artefact` object.
        `add(artefact: Artefact | str, to: Batch.Stage)`: Adds an artefact to the batch at a specified stage.
        `has(artefact: Artefact | str)`: Checks if an artefact is in the batch.
        `move(artefact: Artefact | str, to: Batch.Stage)`: Moves an artefact to a different stage in the batch.
        `where(artefact: Artefact | str)`: Returns the stage of an artefact in the batch.
        `remove(artefact: Artefact | str)`: Removes an artefact from the batch.
        `initJob(jobID: str=None, status: BatchProcessingJob.Status=None, autoStart: bool=False)`: Initializes a processing job for the batch with an optional job ID and status, and optionally starts it.
        `__str__()`: Returns a string representation of the batch, including its ID, user ID, artefacts, job, and creation timestamp.
    
    Static Methods:
        `rawLoad(batchID: str, data: dict, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False)`: Loads a batch from raw data, optionally including user and artefact information.
        `load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False)`: Loads a batch by ID or user ID, optionally including user and artefact information.
        `ref(id: str)`: Returns a reference to the batch in the database.
    
    Example usage:
    ```python
    from models import DI, Batch, BatchArtefact, BatchProcessingJob, User, Artefact
    DI.setup()
    
    user = User.load()[0] # Load the first user available
    arts = Artefact.load()
    
    batch: Batch = Batch(user.id, arts) # Create a new batch, associated with the given user, containing the artefacts. Artefacts are automatically converted to BatchArtefact objects in the 'unprocessed' stage.
    
    batch.loadAllArtefacts(withMetadata=True) # Loads all artefacts in the batch, including their metadata if available. Warning: time- and memory-intensive operation.
    print(list(batch.artefacts.values())[0].artefact) # outputs the Artefact object of the first BatchArtefact reference in the batch
    
    print(batch.where(arts[0])) # outputs: Batch.Stage.UNPROCESSED
    
    newArt = Artefact.load('some_artefact_not_in_batch')
    
    print(batch.has(newArt)) # outputs: False
    batch.add(newArt, Batch.Stage.UNPROCESSED) # constructs BatchArtefact reference to new artefact with the 'unprocessed' stage
    print(batch.has(newArt)) # outputs: True

    print(batch.move(arts[0], Batch.Stage.PROCESSED))
    print(batch.where(arts[0])) # outputs: Batch.Stage.PROCESSED
    
    batch.initJob(jobID='someJobID', autoStart=True) # Initializes a processing job for the batch
    print(batch.job) # outputs BatchProcessingJob instance
    
    batch.job.end() # note: .cancel() auto saves by default
    batch.save() # remember to save after any modifications
    print(batch) # outputs the string representation of the batch with all artefacts and job information
    ```
    """
    
    class Stage(str, Enum):
        """Represents the various stages a batch can be in.
        
        Options:
            UNPROCESSED: The artefact has not been processed yet.
            PROCESSED: The artefact has been processed.
            CONFIRMED: The artefact has been confirmed after processing.
        """
        
        UNPROCESSED = "unprocessed"
        PROCESSED = "processed"
        CONFIRMED = "confirmed"
        
        @staticmethod
        def validateAndReturn(stage: 'Batch.Stage | str') -> 'Batch.Stage | Literal[False]':
            """Validates the given stage and returns it if valid, or False if invalid.

            Args:
                stage (str | Batch.Stage): The stage to validate.

            Returns: `Batch.Stage | Literal[False]`: The validated stage or False if invalid.
            """
            
            if isinstance(stage, Batch.Stage):
                return stage
            elif isinstance(stage, str):
                try:
                    return Batch.Stage(stage.lower())
                except ValueError:
                    return False
            else:
                return False
    
    def __init__(self, userID: str, batchArtefacts: 'List[BatchArtefact | Artefact]'=None, job: 'BatchProcessingJob | None'=None, created: str=None, id: str=None):
        """Initializes a new Batch instance.

        Args:
            userID (str): The ID of the user associated with the batch.
            batchArtefacts (List[BatchArtefact | Artefact], optional): The artefacts to include in the batch. Both `BatchArtefact` and `Artefact` objects are accepted. `Artefact` objects are converted into 'unprocessed' `BatchArtefact` objects. Defaults to None.
            job (BatchProcessingJob | None, optional): The processing job associated with the batch. Defaults to None.
            created (str, optional): The creation timestamp of the batch. Defaults to None.
            id (str, optional): The unique ID of the batch. Defaults to None.

        Raises:
            Exception: If the initialization fails or if the provided artefacts are not of the expected type.
        """
        
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if batchArtefacts is None:
            batchArtefacts = {}
        
        batchArts = {}
        for item in batchArtefacts:
            if isinstance(item, BatchArtefact):
                batchArts[item.artefactID] = item
            elif isinstance(item, Artefact):
                batchArts[item.id] = BatchArtefact(id, item.id, Batch.Stage.UNPROCESSED)
            else:
                raise Exception("BATCH INIT ERROR: Expected item in 'batchArtefacts' to be either BatchArtefact or Artefact, but got '{}'.".format(type(item).__name__))
        
        self.id: str = id
        self.userID: str = userID
        self.artefacts: Dict[str, BatchArtefact] = batchArts
        self.job: BatchProcessingJob | None = job
        self.user: User | None = None
        self.created: str = created
        self.originRef = Batch.ref(self.id)
    
    def save(self, checkIntegrity: bool=True) -> bool:
        """Saves the batch to the database.

        Args:
            checkIntegrity (bool, optional): Whether to check the integrity of the batch before saving. Defaults to True.

        Raises:
            Exception: If the batch does not have a userID set.
            Exception: If the user associated with the batch does not exist.
            Exception: If any artefact in the batch is not a valid BatchArtefact.
            Exception: If the batch's job is not a valid BatchProcessingJob.
            Exception: If a batch artefact's batch ID does not match `self.batchID`.
            Exception: If a batch artefact's artefact ID does not match the expected ID.

        Returns:
            bool: True if the batch was saved successfully.
        """
        
        if checkIntegrity:
            if self.userID is None:
                raise Exception("BATCH SAVE ERROR: Batch does not have a userID set.")
            
            data = User.load(self.userID)
            if data is None:
                raise Exception("BATCH SAVE ERROR: User with ID '{}' does not exist.".format(self.userID))
            del data  # We don't need the user data here, just checking existence
            
            for artefactID, batchArt in self.artefacts.items():
                if not isinstance(batchArt, BatchArtefact):
                    raise Exception("BATCH SAVE ERROR: Artefact with ID '{}' is not a valid BatchArtefact.".format(artefactID))
                if batchArt.batchID != self.id:
                    raise Exception("BATCH SAVE ERROR: Artefact with ID '{}' does not belong to this batch (ID: '{}').".format(artefactID, self.id))
                if batchArt.artefactID != artefactID:
                    raise Exception("BATCH SAVE ERROR: Found artefact ID '{}' in BatchArtefact when '{}' was expected.".format(batchArt.artefactID, artefactID))
                if Batch.Stage.validateAndReturn(batchArt.stage) == False:
                    raise Exception("BATCH SAVE ERROR: Invalid stage '{}' for artefact with ID '{}'.".format(batchArt.stage, artefactID))
            
            if self.job != None and not isinstance(self.job, BatchProcessingJob):
                raise Exception("BATCH SAVE ERROR: 'job' is neither None nor a valid BatchProcessingJob instance.")
        
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "userID": self.userID,
            "artefacts": {id: batchArt.represent() for id, batchArt in self.artefacts.items()},
            'job': self.job.represent() if self.job != None else None,
            "created": self.created
        }
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCH RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCH RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCH RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.id, data).__dict__)
        return True
    
    def __str__(self):
        return """<Batch instance
ID: {}
User ID: {}
User Object:{}
Artefact References:{}
Job:{}
Created: {} />
        """.format(
            self.id,
            self.userID,
            self.user.represent() if isinstance(self.user, User) else " None",
            ("\n---\n- " + ("\n- ".join(str(batchArt) if isinstance(batchArt, BatchArtefact) else "CORRUPTED BATCHARTEFACT AT '{}'".format(artID) for artID, batchArt in self.artefacts.items())) + "\n---") if isinstance(self.artefacts, dict) else " None",
            self.job if isinstance(self.job, BatchProcessingJob) else " None",
            self.created
        )
    
    def getUser(self) -> bool:
        """Loads the user associated with the batch into the `user` attribute.

        Raises:
            Exception: If the `userID` is invalid or if the user cannot be loaded.

        Returns:
            bool: True if the user was successfully loaded, False otherwise.
        """
        
        if not isinstance(self.userID, str):
            raise Exception("BATCH GETUSER ERROR: Invalid value set for 'userID'. Expected a string representing the User ID.")
        
        self.user = User.load(id=self.userID)
        if not isinstance(self.user, User):
            self.user = None
            raise Exception("BATCH GETUSER ERROR: Failed to load User with ID '{}'; response: {}".format(self.userID, self.user))
        
        return True
    
    def loadAllArtefacts(self, withMetadata: bool=False) -> bool:
        """Loads all artefacts associated with the batch.

        Args:
            withMetadata (bool, optional): If True, includes metadata in the artefact loading process. Defaults to False.

        Raises:
            Exception: If the `artefacts` attribute is not a dictionary.
            Exception: If a value in `artefacts` is not a valid `BatchArtefact`.
            Exception: If an artefact cannot be loaded.

        Returns:
            bool: True if all artefacts were successfully loaded.
        """
        
        if not isinstance(self.artefacts, dict):
            raise Exception("BATCH LOADARTEFACTS ERROR: Expected 'artefacts' to be a dictionary, but got '{}'.".format(type(self.artefacts).__name__))
        
        for artefactID, batchArt in self.artefacts.items():
            if not isinstance(batchArt, BatchArtefact):
                raise Exception("BATCH LOADARTEFACTS ERROR: Artefact with ID '{}' is not a valid BatchArtefact.".format(artefactID))
            
            batchArt.getArtefact(includeMetadata=withMetadata)
        
        return True
    
    def get(self, artefact: Artefact | str) -> 'BatchArtefact | None':
        """Retrieves a `BatchArtefact` reference from the batch.

        Args:
            artefact (Artefact | str): The artefact to retrieve, either as an Artefact instance or its ID.

        Returns: `BatchArtefact | None`: The requested artefact if it exists, None otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return self.artefacts.get(artefactID, None)
    
    def add(self, artefact: Artefact | str, to: 'Batch.Stage') -> 'BatchArtefact':
        """Adds an artefact to the batch, creating a `BatchArtefact` reference.

        Args:
            artefact (Artefact | str): The artefact to add, either as an Artefact instance or its ID.
            to (Batch.Stage): The stage of the artefact reference.

        Raises:
            Exception: If the `artefact` value is not valid.
            Exception: If the artefact reference already exists in the batch.

        Returns:
            BatchArtefact: The added artefact reference object.
        """
        
        to = Batch.Stage.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH ADD ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if self.has(artefactID):
            raise Exception("BATCH ADD ERROR: Artefact with ID '{}' already exists in the batch.".format(artefactID))
        
        batchArt = BatchArtefact(self.id, artefactID, to)
        self.artefacts[artefactID] = batchArt
        
        return batchArt
    
    def has(self, artefact: Artefact | str) -> bool:
        """Checks if a reference to an artefact exists in the batch.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact instance or its ID.

        Returns:
            bool: True if the artefact exists in the batch, False otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return artefactID in self.artefacts
    
    def move(self, artefact: Artefact | str, to: 'Batch.Stage') -> bool:
        """Moves an artefact reference to a different stage in the batch.

        Args:
            artefact (Artefact | str): The artefact to move, either as an Artefact instance or its ID.
            to (Batch.Stage): The stage to which the artefact should be moved.

        Raises:
            Exception: If the `artefact` value is not valid.
            Exception: If the artefact reference does not exist in the batch.

        Returns:
            bool: True if the artefact was moved successfully, False otherwise.
        """
        
        to = Batch.Stage.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH MOVE ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if not self.has(artefactID):
            raise Exception("BATCH MOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        batchArt = self.artefacts[artefactID]
        batchArt.stage = to
        
        return True
    
    def where(self, artefact: Artefact | str) -> 'Batch.Stage | None':
        """Retrieves the stage of a specific artefact reference in the batch.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact instance or its ID.

        Returns: `Batch.Stage | None`: The stage of the artefact if it exists in the batch, None otherwise.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return self.artefacts[artefactID].stage if self.has(artefactID) else None
    
    def remove(self, artefact: Artefact | str) -> bool:
        """Removes an artefact reference from the batch.

        Args:
            artefact (Artefact | str): The artefact to remove, either as an Artefact instance or its ID.

        Returns:
            bool: True if the artefact was removed successfully.
        """
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        self.artefacts.pop(artefactID, None)
        return True
    
    def initJob(self, jobID: str=None, status: 'BatchProcessingJob.Status'=None, autoStart: bool=False) -> 'BatchProcessingJob':
        """Initializes a new `BatchProcessingJob` under the `job` attribute.

        Args:
            jobID (str, optional): The ID of the job. Defaults to None.
            status (BatchProcessingJob.Status, optional): The status of the job. Defaults to `BatchProcessingJob.Status.PENDING`.
            autoStart (bool, optional): Whether to start the job automatically. Defaults to False.

        Raises:
            Exception: If the `status` value is provided but not valid.

        Returns:
            BatchProcessingJob: The initialized batch processing job object.
        """
        
        if status is None:
            status = BatchProcessingJob.Status.PENDING
        
        status = BatchProcessingJob.Status.validateAndReturn(status)
        if status == False:
            raise Exception("BATCH INITJOB ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        if isinstance(self.job, BatchProcessingJob):
            if self.job.status == BatchProcessingJob.Status.PROCESSING or self.job.ended is None:
                Logger.log("BATCH INITJOB WARNING: Overwriting existing job with processing status or no 'ended' value with new job with ID: '{}' and status: '{}'.".format(jobID, status))
        
        self.job = BatchProcessingJob(self.id, jobID, status)
        
        if autoStart:
            self.job.start(jobID)
        
        return self.job
    
    def __str__(self):
        return """<Batch instance
ID: {}
User ID: {}
User Object: {}
Artefacts:{}
Job: {}
Created: {} />
        """.format(
            self.id,
            self.userID,
            self.user.represent() if self.user is not None else " None",
            ("\n---\n- " + ("\n- ".join(str(batchArt) if isinstance(batchArt, BatchArtefact) else "CORRUPTED BATCHARTEFACT AT '{}'".format(artID) for artID, batchArt in self.artefacts.items())) + "\n---") if isinstance(self.artefacts, dict) else " None",
            self.job,
            self.created
        )
    
    @staticmethod
    def rawLoad(batchID: str, data: dict, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch':
        """Constructs a `Batch` from raw dictionary data.

        Args:
            batchID (str): The ID of the batch.
            data (dict): The raw data for the batch.
            withUser (bool, optional): Whether to automatically load the associated `User` object. Defaults to False.
            withArtefacts (bool, optional): Whether to automatically load `Artefact` objects internally in `BatchArtefact` references. Defaults to False.
            metadataRequired (bool, optional): Whether metadata should be loaded in the Artefact loading process. Defaults to False. Effective if `withArtefacts` is True. Warning: may cause high memory usage.

        Returns:
            Batch: The loaded batch object.
        """
        
        requiredParams = ['userID', 'artefacts', 'job', 'created']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'artefacts':
                    data[reqParam] = {}
                else:
                    data[reqParam] = None
        
        if not isinstance(data['artefacts'], dict):
            data['artefacts'] = {}
        
        arts = []
        for artefactID, batchArtData in data['artefacts'].items():
            if isinstance(batchArtData, dict):
                arts.append(BatchArtefact.rawLoad(batchID, artefactID, batchArtData))
        
        job = None
        if isinstance(data['job'], dict):
            job = BatchProcessingJob.rawLoad(batchID, data['job'])
        
        batch = Batch(
            userID=data.get('userID'),
            batchArtefacts=arts,
            job=job,
            created=data.get('created'),
            id=batchID
        )
        
        if withUser:
            batch.getUser()
        if withArtefacts:
            batch.loadAllArtefacts(withMetadata=metadataRequired)
        
        return batch
    
    @staticmethod
    def load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch | Dict[Batch] | None':
        """Loads a batch from the database.

        Args:
            id (str, optional): The ID of the batch to load. Defaults to None.
            userID (str, optional): Produces a list of batches associated with the user ID. Defaults to None.
            withUser (bool, optional): Whether to automatically load the associated `User` object. Defaults to False.
            withArtefacts (bool, optional): Whether to automatically load `Artefact` objects internally in `BatchArtefact` references. Defaults to False.
            metadataRequired (bool, optional): Whether metadata should be loaded in the Artefact loading process. Defaults to False. Effective if `withArtefacts` is True. Warning: may cause high memory usage.

        Returns: `Batch | Dict[Batch] | None`: The loaded batch or list of batches, or None if not found.
        """
        
        if id != None:
            data = DI.load(Batch.ref(id))
            if data is None:
                if userID is not None:
                    return Batch.load(userID=userID, withUser=withUser, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            batch = Batch.rawLoad(id, data, withUser, withArtefacts, metadataRequired)
            
            return batch
        else:
            data = DI.load(Ref("batches"))
            if data is None:
                return []
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Failed to load dictionary batches data; response: {}".format(data))
            
            batches: Dict[str, Batch] = {}
            for id in data:
                if isinstance(data[id], dict):
                    batches[id] = Batch.rawLoad(id, data[id], withUser, withArtefacts, metadataRequired)
            
            if userID is None:
                return list(batches.values())
            
            userBatches: List[Batch] = []
            for batchID in batches:
                if batches[batchID].userID == userID:
                    userBatches.append(batches[batchID])
            
            return userBatches
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("batches", id)

class BatchProcessingJob(DIRepresentable):
    """Represents a batch processing job. Contains key attributes such as job ID, status and timestamps for when the job started and ended. A `DIRepresentable` model.

    Attributes:
        batchID (str): The ID of the batch this job is associated with.
        jobID (str | None): The ID of the job, if available.
        status (BatchProcessingJob.Status): The current status of the job.
        started (str | None): The timestamp when the job started, if available.
        ended (str | None): The timestamp when the job ended, if available.
    
    Methods:
        `save()`: Saves the job data to the database.
        `represent()`: Returns a dictionary representation of the job.
        `start(jobID: str=None)`: Starts the job, setting its status to PROCESSING and updating the started timestamp. Optionally sets a job ID.
        `end()`: Ends the job, setting its status to COMPLETED and updating the ended timestamp.
        `cancel(autoSave: bool=True)`: Cancels the job, setting its status to CANCELLED and updating the ended timestamp. Optionally saves the job data.
        `reload()`: Reloads the job data from the database.
        `setStatus(newStatus: BatchProcessingJob.Status)`: Sets the job's status to a new value, validating it first.
        `__str__()`: Returns a string representation of the job.
    
    Static Methods:
        `rawLoad(batchID: str, data: dict)`: Constructs a `BatchProcessingJob` from raw data, validating required parameters and setting defaults if necessary.
        `load(batch: Batch | str)`: Loads a `BatchProcessingJob` for a given batch, either by batch ID or `Batch` object. Returns None if no job is found.
        `ref(batchID: str)`: Returns a reference to the job in the database.
    
    Example usage:
    ```python
    import time
    from models import DI, Batch, BatchProcessingJob
    DI.setup()
    
    job = BatchProcessingJob.load(batchID='some_batch_id')
    job.start()
    
    time.sleep(2)  # Simulate processing time
    
    job.end()
    job.save()  # Save the job data to the database
    print(job)  # Outputs the string representation of the job
    ```
    """
    
    class Status(str, Enum):
        """Represents the various statuses a batch processing job can have.
        
        Options:
            PENDING: The job is waiting to be processed.
            PROCESSING: The job is currently being processed.
            COMPLETED: The job has been successfully completed.
            CANCELLED: The job has been cancelled.
        """
        
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        CANCELLED = "cancelled"
        
        @staticmethod
        def validateAndReturn(status) -> 'BatchProcessingJob.Status | Literal[False]':
            if isinstance(status, BatchProcessingJob.Status):
                return status
            elif isinstance(status, str):
                try:
                    return BatchProcessingJob.Status(status.lower())
                except ValueError:
                    return False
            else:
                return False
    
    def __init__(self, batch: Batch | str, jobID: str | None, status: 'BatchProcessingJob.Status', started: str | None=None, ended: str | None=None):
        """Initializes a new batch processing job.

        Args:
            batch (Batch | str): The batch object or ID associated with the job.
            jobID (str | None): Typically the thread ID of the job.
            status (BatchProcessingJob.Status): The current status of the job.
            started (str | None, optional): The timestamp when the job was started. Defaults to None.
            ended (str | None, optional): The timestamp when the job was ended. Defaults to None.

        Raises:
            Exception: If the provided status is invalid.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        status = BatchProcessingJob.Status.validateAndReturn(status)
        if status == False:
            raise Exception("BATCHPROCESSINGJOB INIT ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        self.batchID: str = batchID
        self.jobID: str | None = jobID
        self.status: BatchProcessingJob.Status = status
        self.started: str | None = started
        self.ended: str | None = ended
        self.originRef = BatchProcessingJob.ref(batchID)
    
    def represent(self):
        return {
            "jobID": self.jobID,
            "status": self.status.value,
            "started": self.started,
            "ended": self.ended
        }
    
    def save(self):
        return DI.save(self.represent(), self.originRef)
    
    def start(self, jobID: str=None) -> bool:
        """Starts the batch processing job, setting its status to PROCESSING and updating the started timestamp.

        Args:
            jobID (str, optional): The unique identifier for the job. Defaults to None.

        Returns:
            bool: True if the job was successfully started.
        """
        
        if self.status != BatchProcessingJob.Status.PENDING:
            Logger.log("BATCHPROCESSINGJOB START WARNING: Starting job with ID '{}' that is not in PENDING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        if self.jobID != None:
            self.jobID = jobID
        self.status = BatchProcessingJob.Status.PROCESSING
        self.started = Universal.utcNowString()
        
        return True
    
    def end(self) -> bool:
        """Ends the batch processing job, setting its status to COMPLETED and updating the ended timestamp.

        Returns:
            bool: True if the job was successfully ended.
        """
        
        if self.status != BatchProcessingJob.Status.PROCESSING:
            Logger.log("BATCHPROCESSINGJOB END WARNING: Ending job with ID '{}' that is not in PROCESSING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        self.status = BatchProcessingJob.Status.COMPLETED
        self.ended = Universal.utcNowString()
        
        return True
    
    def cancel(self, autoSave: bool=True) -> bool:
        """Cancels the batch processing job, setting its status to CANCELLED and updating the ended timestamp.

        Args:
            autoSave (bool, optional): Whether to automatically save the job state. Defaults to True.

        Returns:
            bool: True if the job was successfully cancelled.
        """
        
        if self.status != BatchProcessingJob.Status.PROCESSING:
            Logger.log("BATCHPROCESSINGJOB CANCEL WARNING: Cancelling job with ID '{}' that is not in PROCESSING status; current status: '{}'.".format(self.jobID, self.status.value))
        
        self.status = BatchProcessingJob.Status.CANCELLED
        self.ended = Universal.utcNowString()
        
        if autoSave:
            return self.save()
        
        return True
    
    def fetchLatestStatus(self) -> 'BatchProcessingJob.Status | None':
        self.status = BatchProcessingJob.getStatus(self.batchID)
        return self.status
    
    def computeDuration(self) -> float | None:
        if not isinstance(self.started, str) or not isinstance(self.ended, str):
            return None
        try:
            start = Universal.fromUTC(self.started)
            end = Universal.fromUTC(self.ended)
            return (end - start).total_seconds()
        except Exception as e:
            Logger.log("BATCHPROCESSINGJOB COMPUTEDURATION ERROR: {}".format(e))
            return None
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCHPROCESSINGJOB RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        self.__dict__.update(self.rawLoad(self.batchID, data).__dict__)
        return True
    
    def setStatus(self, newStatus: 'BatchProcessingJob.Status') -> bool:
        """Sets the status of the batch processing job.

        Args:
            newStatus (BatchProcessingJob.Status): The new status to set.

        Raises:
            Exception: If the provided status is invalid.

        Returns:
            bool: True if the status was successfully updated.
        """
        
        newStatus = BatchProcessingJob.Status.validateAndReturn(newStatus)
        if newStatus == False:
            raise Exception("BATCHPROCESSINGJOB SETSTATUS ERROR: Invalid status provided; expected 'BatchProcessingJob.Status' or string representation of it.")
        
        self.status = newStatus
        return True
    
    def __str__(self):
        return "BatchProcessingJob(batchID='{}', jobID='{}', status='{}', started='{}', ended='{}')".format(
            self.batchID,
            self.jobID,
            self.status.value,
            self.started,
            self.ended
        )

    @staticmethod
    def rawLoad(batchID: str, data: dict) -> 'BatchProcessingJob':
        """Loads a batch processing job from the provided data dictionary.

        Args:
            batchID (str): The ID of the batch this job is associated with.
            data (dict): The data dictionary containing job information.

        Returns:
            BatchProcessingJob: The job object.
        """
        
        reqParams = ['jobID', 'status', 'started', 'ended']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'status':
                    data[reqParam] = BatchProcessingJob.Status.PENDING.value
                else:
                    data[reqParam] = None
        
        status = BatchProcessingJob.Status.validateAndReturn(data['status'])
        if status == False:
            status = BatchProcessingJob.Status.PENDING
        
        return BatchProcessingJob(
            batch=batchID,
            jobID=data.get('jobID'),
            status=status,
            started=data.get('started'),
            ended=data.get('ended')
        )
    
    @staticmethod
    def load(batch: Batch | str) -> 'BatchProcessingJob | None':
        """Loads a batch processing job from the provided batch identifier.

        Args:
            batch (Batch | str): The batch object or ID this job is associated with.

        Raises:
            Exception: If the provided batch is invalid.
            Exception: If the job cannot be loaded.

        Returns: `BatchProcessingJob | None`: The loaded job object or None if not found.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        
        data = DI.load(BatchProcessingJob.ref(batchID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("BATCHPROCESSINGJOB LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return BatchProcessingJob.rawLoad(batchID, data)
    
    @staticmethod
    def getStatus(batch: Batch | str) -> 'BatchProcessingJob.Status | None':
        """Gets the status of the batch processing job.

        Args:
            batch (Batch | str): The batch object or ID to retrieve the job status for.

        Raises:
            Exception: If any error occurs during status retrieval and validation.

        Returns: `BatchProcessingJob.Status | None`: The status of the job if it exists, None otherwise.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        if not isinstance(batchID, str):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Expected 'batch' to be a Batch object or a string representing the batch ID, but got '{}'.".format(type(batch).__name__))
        
        data = DI.load(Ref("batches", batchID, "job", "status"))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, str):
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Unexpected DI load response format; response: {}".format(data))
        
        status = BatchProcessingJob.Status.validateAndReturn(data)
        if status == False:
            raise Exception("BATCHPROCESSINGJOB GETSTATUS ERROR: Invalid status value '{}' returned from DI.".format(data))
        
        return status
    
    @staticmethod
    def ref(batchID: str) -> Ref:
        return Ref("batches", batchID, "job")

class BatchArtefact(DIRepresentable):
    """Represents an artefact reference within a batch. A junction model between `Batch` and `Artefact`, this model contains key attributes
    that link the two entities together with additional information.
    
    Attributes:
        batchID (str): The ID of the batch this artefact is associated with.
        artefactID (str): The ID of the artefact this reference points to.
        stage (Batch.Stage): The current processing stage of the artefact within the batch.
        processedDuration (str | None): The duration for which the artefact has been processed, if applicable.
        processedTime (str | None): The timestamp when the artefact was processed, if applicable.
        processingError (str | None): Any error that occurred during the processing of the artefact, if applicable.
        artefact (Artefact | None): A convenience attribute that holds the `Artefact` object if it has been loaded.
        originRef (Ref): The reference to the artefact in the database.
    
    Methods:
        `save()`: Saves the artefact reference to the database.
        `represent()`: Returns a dictionary representation of the artefact reference.
        `reload()`: Reloads the artefact reference from the database.
        `getArtefact(includeMetadata: bool=False)`: Loads the associated `Artefact` object into the `artefact` attribute, optionally including metadata.
        `setStage(newStage: Batch.Stage)`: Sets the stage of the artefact reference to a new value, validating it first
        `__str__()`: Returns a string representation of the artefact reference.
    
    Static Methods:
        `rawLoad(batchID: str, artefactID: str, data: dict, withArtefact: bool=False, metadataRequired: bool=False)`: Constructs a `BatchArtefact` from raw data, validating required parameters and setting defaults if necessary.
        `load(batch: Batch | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False)`: Loads a `BatchArtefact` for a given batch and artefact, either by IDs or objects. Returns None if not found.
        `ref(batchID: str, artefactID: str)`: Returns a reference to the artefact in the database.
    
    Example usage:
    ```python
    from services import Universal
    from models import DI, Batch, Artefact, BatchArtefact
    DI.setup()
    
    batch = Batch.load(id='some_batch_id')
    artefact = Artefact.load(id='some_artefact_id')
    
    batchArtefact = BatchArtefact(batch=batch, artefact=artefact, stage=Batch.Stage.PROCESSED)
    
    batchArtefact.processedDuration = "32.456"
    batchArtefact.processedTime = Universal.utcNowString()
    batchArtefact.save() # Save the artefact reference to the database
    
    print(batchArtefact)  # Outputs the string representation of the artefact reference
    ```
    """
    
    def __init__(self, batch: Batch | str, artefact: Artefact | str, stage: Batch.Stage, processedDuration: str | None=None, processedTime: str | None=None, processingError: str | None=None):
        """Initializes a BatchArtefact instance.

        Args:
            batch (Batch | str): The batch object or ID this artefact is associated with.
            artefact (Artefact | str): The artefact object or ID this reference is associated with.
            stage (Batch.Stage): The current processing stage of the artefact.
            processedDuration (str | None, optional): The duration for which the artefact has been processed, if applicable. Defaults to None.
            processedTime (str | None, optional): The timestamp when the artefact was processed, if applicable. Defaults to None.
            processingError (str | None, optional): Any error that occurred during the processing of the artefact, if applicable. Defaults to None.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        stage = Batch.Stage.validateAndReturn(stage)
        if stage == False:
            raise Exception("BATCHARTEFACT INIT ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        self.batchID: str = batchID
        self.artefactID: str = artefactID
        self.stage: Batch.Stage = stage
        self.processedDuration: str | None = processedDuration
        self.processedTime: str | None = processedTime
        self.processingError: str | None = processingError
        self.artefact: Artefact | None = artefact if isinstance(artefact, Artefact) else None
        self.originRef = BatchArtefact.ref(batchID, artefactID)
    
    def represent(self):
        return {
            "stage": self.stage.value,
            "processedDuration": self.processedDuration,
            "processedTime": self.processedTime,
            "processingError": self.processingError
        }
    
    def save(self):
        return DI.save(self.represent(), self.originRef)
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("BATCHARTEFACT RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("BATCHARTEFACT RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("BATCHARTEFACT RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.batchID, self.artefactID, data).__dict__)
        return True
    
    def getArtefact(self, includeMetadata: bool=False) -> bool:
        """Retrieves the associated `Artefact` object into `artefact` for convenience.

        Args:
            includeMetadata (bool, optional): Whether to include metadata in the retrieval. Defaults to False.

        Raises:
            Exception: If the `artefactID` is not a string or if the artefact cannot be loaded.

        Returns:
            bool: True if the artefact was successfully retrieved.
        """
        
        if not isinstance(self.artefactID, str):
            raise Exception("BATCHARTEFACT GETARTEFACT ERROR: Invalid value set for 'artefactID'. Expected a string representing the Artefact ID.")
        
        self.artefact = Artefact.load(id=self.artefactID, includeMetadata=includeMetadata)
        if not isinstance(self.artefact, Artefact):
            self.artefact = None
            raise Exception("BATCHARTEFACT GETARTEFACT ERROR: Failed to load Artefact with ID '{}'; response: {}".format(self.artefactID, self.artefact))
        
        return True
    
    def setStage(self, newStage: Batch.Stage) -> bool:
        """Sets the processing stage of the artefact reference.

        Args:
            newStage (Batch.Stage): The new stage to set.

        Raises:
            Exception: If the provided stage is invalid.

        Returns:
            bool: True if the stage was successfully set.
        """
        
        newStage = Batch.Stage.validateAndReturn(newStage)
        if newStage == False:
            raise Exception("BATCHARTEFACT SETSTAGE ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        self.stage = newStage
        return True
    
    def __str__(self):
        return "BatchArtefact(batchID='{}', artefactID='{}', stage='{}', processedDuration='{}', processedTime='{}', processingError='{}', artefact={})".format(
            self.batchID,
            self.artefactID,
            self.stage.value,
            self.processedDuration,
            self.processedTime,
            self.processingError,
            self.artefact if self.artefact != None else None
        )
    
    @staticmethod
    def rawLoad(batchID: str, artefactID: str, data: dict, withArtefact: bool=False, metadataRequired: bool=False) -> 'BatchArtefact':
        """Loads a BatchArtefact from raw dictionary data.

        Args:
            batchID (str): The ID of the batch this artefact is associated with.
            artefactID (str): The ID of the referred artefact.
            data (dict): The raw batch artefact reference information.
            withArtefact (bool, optional): Whether to automatically load the `Artefact` object internally. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is included in the `Artefact` loading process. Defaults to False. Effective if `withArtefact` is True. Warning: may cause high memory usage.

        Returns:
            BatchArtefact: The loaded BatchArtefact instance.
        """
        
        reqParams = ['stage', 'processedDuration', 'processedTime', 'processingError']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'stage':
                    data[reqParam] = Batch.Stage.UNPROCESSED.value
                else:
                    data[reqParam] = None
        
        stage = Batch.Stage.validateAndReturn(data['stage'])
        if stage == False:
            stage = Batch.Stage.UNPROCESSED
        
        batchArt = BatchArtefact(
            batch=batchID,
            artefact=artefactID,
            stage=stage,
            processedDuration=data.get('processedDuration'),
            processedTime=data.get('processedTime'),
            processingError=data.get('processingError')
        )
        
        if withArtefact:
            batchArt.getArtefact(includeMetadata=metadataRequired)
        
        return batchArt
    
    @staticmethod
    def load(batch: Batch | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False) -> 'BatchArtefact | None':
        """Loads a BatchArtefact instance.

        Args:
            batch (Batch | str): The batch object or ID associated with the artefact reference.
            artefact (Artefact | str): The artefact object or ID associated with the artefact reference.
            withArtefact (bool, optional): Whether to automatically load the `Artefact` object internally. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is included in the `Artefact` loading process. Defaults to False. Effective if `withArtefact` is True. Warning: may cause high memory usage.

        Returns: `BatchArtefact | None`: The loaded BatchArtefact instance or None if not found.
        """
        
        batchID = batch.id if isinstance(batch, Batch) else batch
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        data = DI.load(BatchArtefact.ref(batchID, artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("BATCHARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return BatchArtefact.rawLoad(batchID, artefactID, data, withArtefact, metadataRequired)
    
    @staticmethod
    def ref(batchID: str, artefactID: str) -> Ref:
        return Ref("batches", batchID, "artefacts", artefactID)

class User(DIRepresentable):
    """User model representation.
    
    Attributes:
        id (str): Unique identifier for the user.
        username (str): The username of the user.
        email (str): The email of the user.
        pwd (str): The password of the user.
        authToken (str, optional): The auth token of the user. Defaults to None.
        superuser (bool, optional): Whether the user is a superuser. Defaults to False
        lastLogin (str, optional): The last login timestamp of the user. Defaults to None.
        created (str, optional): The creation timestamp of the user. Defaults to None.
    
    Sample usage:
    ```python
    from models import DI, User
    DI.setup()
    
    user = User("johndoe", "john@doe.com", "securepwd")
    user.authToken = "123456"
    user.save()
    
    loadedUser = User.load(username="johndoe")
    if not isinstance(loadedUser, User):
        raise Exception("Failed to load user - they may not exist. Response: {}".format(loadedUser))
    
    print("User data:")
    print(loadedUser)
    ```
    """
    
    def __init__(self, username: str, email: str, pwd: str, authToken: str=None, superuser: bool=False, lastLogin: str=None, created: str=None, id: str=None):
        """User model representation.

        Args:
            username (str): The username of the user.
            email (str): The email of the user.
            pwd (str): The password of the user.
            authToken (str, optional): The auth token of the user. Defaults to None.
            superuser (bool, optional): Whether the user is a superuser. Defaults to False.
            lastLogin (str, optional): The last login timestamp of the user. Defaults to None.
            created (str, optional): The creation timestamp of the user. Defaults to None. If not provided, it will be set to the current UTC time in ISO format.
            id (str, optional): The unique identifier of the user. Defaults to None. Auto-generated if not provided.
        """
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        
        self.id = id
        self.username = username
        self.email = email
        self.pwd = pwd
        self.authToken = authToken
        self.superuser = superuser
        self.lastLogin = lastLogin
        self.created = created
        self.originRef = User.ref(id)
    
    def represent(self) -> dict:
        return {
            "username": self.username,
            "email": self.email,
            "pwd": self.pwd,
            "authToken": self.authToken,
            "superuser": self.superuser,
            "lastLogin": self.lastLogin,
            "created": self.created
        }
    
    def __str__(self):
        return "User(id='{}', username='{}', email='{}', pwd='{}', authToken='{}', superuser={}, lastLogin='{}', created='{}')".format(
            self.id,
            self.username,
            self.email,
            self.pwd,
            self.authToken,
            self.superuser,
            self.lastLogin,
            self.created
        )
    
    def save(self, checkSuperuserIntegrity: bool=True):
        """Saves the user to the database.

        Args:
            checkSuperuserIntegrity (bool, optional): Whether to check for superuser integrity. Defaults to True.

        Raises:
            Exception: If an error occurs during saving.
            Exception: If more than one superuser exists.

        Returns:
            bool: Almost exclusively returns `True`.
        """
        
        if self.superuser == True and checkSuperuserIntegrity:
            allUsers = User.load()
            if not isinstance(allUsers, list):
                raise Exception("USER SAVE ERROR: Unexpected User load response format; response: {}".format(allUsers))
            
            for user in allUsers:
                if user.id != self.id and user.superuser == True:
                    raise Exception("USER SAVE ERROR: Cannot save superuser user when another superuser already exists.")
        
        convertedData = self.represent()
        return DI.save(convertedData, self.originRef)
    
    def reload(self):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("USER RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("USER RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("USER RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    @staticmethod
    def rawLoad(data: dict, userID: str | None=None) -> 'User':
        requiredParams = ['username', 'email', 'pwd', 'authToken', 'superuser', 'lastLogin', 'created']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam in []: # add any required params that should be empty dictionaries
                    data[reqParam] = {}
                elif reqParam in []: # add any required params that should be empty lists
                    data[reqParam] = []
                elif reqParam in ['superuser']:
                    data[reqParam] = False
                else:
                    data[reqParam] = None
        
        if not isinstance(data['superuser'], bool):
            data['superuser'] = False
        
        return User(
            username=data['username'],
            email=data['email'],
            pwd=data['pwd'],
            authToken=data['authToken'],
            superuser=data['superuser'],
            lastLogin=data['lastLogin'],
            created=data['created'],
            id=userID
        )
    
    @staticmethod
    def load(id: str=None, username: str=None, email: str=None, authToken: str=None) -> 'User | List[User] | None':
        """Loads a user from the database.

        Args:
            id (str, optional): The ID of the user to load. Defaults to None.
            username (str, optional): The username of the user to load. Defaults to None.
            email (str, optional): The email of the user to load. Defaults to None.
            authToken (str, optional): The auth token of the user to load. Defaults to None.

        Raises:
            Exception: If an error occurs during loading.
            Exception: If the user is not found.
            Exception: If the user data is invalid.
            Exception: If multiple users match the criteria.

        Returns:
            User | List[User] | None: The loaded user(s) or None if not found.
        """
        
        if id != None:
            data = DI.load(User.ref(id))
            if isinstance(data, DIError):
                raise Exception("USER LOAD ERROR: DIError occurred: {}".format(data))
            if data == None:
                return None
            if not isinstance(data, dict):
                raise Exception("USER LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return User.rawLoad(data, id)
        else:
            data = DI.load(Ref("users"))
            if data == None:
                return []
            if isinstance(data, DIError):
                raise Exception("USER LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("USER LOAD ERROR: Failed to load dictionary users data; response: {}".format(data))

            users: Dict[str, User] = {}
            for id in data:
                if isinstance(data[id], dict):
                    users[id] = User.rawLoad(data[id], id)

            if username == None and email == None and authToken == None:
                return list(users.values())

            for userID in users:
                targetUser = users[userID]
                if username != None and targetUser.username == username:
                    return targetUser
                elif email != None and targetUser.email == email:
                    return targetUser
                elif authToken != None and targetUser.authToken == authToken:
                    return targetUser

            return None
    
    @staticmethod
    def getSuperuser() -> 'User | None':
        """ Retrieves the superuser from the database.

        Raises:
            Exception: If more than one superuser exists.
            Exception: If an error occurs during loading.

        Returns:
            User | None: The superuser if found, otherwise None.
        """
        
        allUsers = User.load()
        if not isinstance(allUsers, list):
            raise Exception("USER GETSUPERUSER ERROR: Unexpected User load response format; response: {}".format(allUsers))
        
        su = None
        for user in allUsers:
            if user.superuser == True:
                if su is not None:
                    raise Exception("USER GETSUPERUSER ERROR: More than one superuser found; cannot determine which to return.")
                su = user

        return su

    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("users", id)

class Category(DIRepresentable):
    """
    Represents a collection of artefacts grouped under a specific category.

    Attributes:
        id (str): Unique identifier for the category.
        name (str): Name of the category.
        description (str): Description of the category.
        members (Dict[str, CategoryArtefact]): Dictionary mapping artefact IDs to CategoryArtefact objects.
        created (str): ISO timestamp of category creation.
        originRef (Ref): Reference object for database operations.

    Methods:
        represent(): Returns a dictionary representation of the category for serialization.
        save(): Saves the category to the database.
        reload(): Reloads the category data from the database.
        add(): Adds an artefact to the category.
        remove(): Removes an artefact from the category.
        has(): Checks if an artefact is in the category.
        get(): Retrieves the CategoryArtefact for a given artefact.
        cleanMembers(): Removes invalid or missing artefacts from members in a safe manner.
        loadAllArtefacts(): Loads all artefact objects for the category.
        rawLoad(): Loads a Category from raw dictionary data.
        load(): Loads a category by ID, name, or member ID.
        ref(id): Returns the database reference for the category.

    Sample usage:
    ```python
    from models import DI, Category, Artefact
    DI.setup()

    # Create a new category
    cat = Category(name="Paintings", description="All painting artefacts")
    cat.save()

    # Add an artefact to the category
    artefact = Artefact.load(id="artefact123")
    cat.add(artefact, reason="Featured in exhibition")
    cat.save()

    # Load and inspect a category
    loaded_cat = Category.load(name="Paintings", withArtefacts=True, metadataRequired=True) # loads 'Artefact' objects, which internally loads 'Metadata', internally in 'CategoryArtefact'
    print(loaded_cat.represent())
    ```

    Notes:
        - All artefacts added to a category must exist in the database.
        - The `members` attribute is a dictionary mapping artefact IDs to CategoryArtefact objects.
        - Use `save()` after modifying a category to persist changes.
        - The class supports both direct artefact objects and artefact IDs for member management.
    """
    
    def __init__(self, name: str, description: str, members: 'Dict[str, CategoryArtefact]'=None, created: str=None, id: str=None):
        """Initialize a new Category instance.

        Args:
            name (str): The name of the category.
            description (str): A brief description of the category.
            members (Dict[str, CategoryArtefact], optional): A dictionary of artefacts belonging to the category. Defaults to None.
            created (str, optional): ISO timestamp of category creation. Defaults to None.
            id (str, optional): Unique identifier for the category. Defaults to None.
        """
        
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if members is None:
            members = {}
        
        self.id = id
        self.name = name
        self.description = description
        self.members = members
        self.created = created
        self.originRef = Category.ref(id)
    
    def represent(self) -> dict:
        """Returns a dictionary representation of the category for serialization.

        Returns:
            dict: A dictionary containing the category's attributes.
        """
        
        return {
            "name": self.name,
            "description": self.description,
            "members": {artefactID: member.represent() for artefactID, member in self.members.items()},
            "created": self.created
        }
    
    def save(self, checkIntegrity: bool=True):
        """Saves the category to the database. Integrity check enforces unique names and integrity of 'members' attribute.
        
        Args:
            checkIntegrity (bool, optional): Whether to perform integrity checks before saving. Defaults to True.

        Returns:
            bool: True if the save operation was successful, False otherwise.
        """
        
        if checkIntegrity:
            if not isinstance(self.members, dict):
                raise Exception("CATEGORY SAVE ERROR: 'members' must be a dictionary of CategoryArtefact objects.")
            
            for artefactID, member in self.members.items():
                if not isinstance(member, CategoryArtefact):
                    raise Exception("CATEGORY SAVE ERROR: 'members' must contain CategoryArtefact objects; found type '{}' for artefact ID '{}'.".format(type(member), artefactID))
                if member.categoryID != self.id:
                    raise Exception("CATEGORY SAVE ERROR: CategoryArtefact with ID '{}' does not belong to this category (expected category ID '{}').".format(artefactID, self.id))
            
            cat = Category.load(name=self.name)
            if isinstance(cat, Category) and cat.id != self.id:
                raise Exception("CATEGORY SAVE ERROR: 'name' attribute uniqueness constraint fails due to category with ID '{}' with matching name '{}'.".format(cat.id, cat.name))

        return DI.save(self.represent(), self.originRef)
    
    def reload(self):
        """Reloads the category data from the database.

        Raises:
            Exception: If a DIError occurs during loading.
            Exception: If no data is found for the category.
            Exception: If the data format is unexpected.

        Returns:
            bool: True if the reload operation was successful, False otherwise.
        """
        
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("CATEGORY RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("CATEGORY RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("CATEGORY RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    def add(self, artefact: Artefact | str, reason: str, loadArtefactObject: bool=False, metadataRequired: bool=False) -> bool:
        """Adds an artefact to the category.

        Args:
            artefact (Artefact | str): The artefact to add, either as an Artefact object or a string ID.
            reason (str): The reason for adding the artefact.
            checkIntegrity (bool, optional): Whether to check the integrity of the artefact. Defaults to True.
            loadArtefactObject (bool, optional): Whether to load the artefact object. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required. Defaults to False.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.
            Exception: If 'reason' is not a string.
            Exception: If the artefact already exists in the category.
            Exception: If the artefact cannot be loaded when checkIntegrity is True.
            Exception: If an artefact does not exist when loadArtefactObject is True.

        Returns:
            bool: True if the artefact was added successfully, False otherwise.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY ADD ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        if not isinstance(reason, str):
            raise Exception("CATEGORY ADD ERROR: 'reason' must be a string.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        if artefactID in self.members:
            raise Exception("CATEGORY ADD ERROR: Artefact with ID '{}' already exists.".format(artefactID))
        
        catArt = CategoryArtefact(
            category=self.id,
            artefact=artefact,
            reason=reason
        )
        
        if loadArtefactObject:
            catArt.getArtefact(includeMetadata=metadataRequired)
        
        self.members[artefactID] = catArt
        
        return True
    
    def remove(self, artefact: Artefact | str) -> bool:
        """Removes an artefact from the category.

        Args:
            artefact (Artefact | str): The artefact to remove, either as an Artefact object or a string ID.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.
            Exception: If the artefact does not exist in the category.

        Returns:
            bool: True if the artefact was removed successfully.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY REMOVE ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        if artefactID not in self.members:
            raise Exception("CATEGORY REMOVE ERROR: Artefact with ID '{}' does not exist in the category.".format(artefactID))
        
        del self.members[artefactID]
        return True

    def has(self, artefact: Artefact | str) -> bool:
        """Checks if the category contains a specific artefact.

        Args:
            artefact (Artefact | str): The artefact to check, either as an Artefact object or a string ID.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.

        Returns:
            bool: True if the artefact is in the category, False otherwise.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY HAS ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        return artefactID in self.members
    
    def get(self, artefact: Artefact | str, default: None=None) -> 'CategoryArtefact | None':
        """Retrieves an artefact from the category.

        Args:
            artefact (Artefact | str): The artefact to retrieve, either as an Artefact object or a string ID.
            default (None, optional): The value to return if the artefact is not found. Defaults to None.

        Raises:
            Exception: If 'artefact' is not an Artefact object or a string.

        Returns:
            CategoryArtefact | None: The requested artefact, or the default value if not found.
        """
        
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY GET ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        return self.members.get(artefactID, default)
    
    def cleanMembers(self) -> bool:
        """Cleans up the members of the category by removing any invalid artefacts.

        Returns:
            bool: True if the cleanup was successful.
        """
        
        for artefactID in list(self.members.keys()):
            if not isinstance(self.members[artefactID], CategoryArtefact):
                del self.members[artefactID]
                continue
            
            if not self.members[artefactID].getArtefact(safeFail=True):
                del self.members[artefactID]
            
        return True
    
    def loadAllArtefacts(self, metadataRequired: bool=False) -> bool:
        """Loads all artefacts in the category.

        Args:
            metadataRequired (bool, optional): Whether to include metadata in the loading process. Defaults to False.

        Raises:
            Exception: If 'self.members' is not a dictionary.
            Exception: If any member is not a CategoryArtefact.

        Returns:
            bool: True if all artefacts were loaded successfully.
        """
        
        if not isinstance(self.members, dict):
            raise Exception("CATEGORY LOADALLARTEFACTS ERROR: 'members' must be a dictionary of CategoryArtefact objects.")
        
        for artefactID in self.members:
            if isinstance(self.members[artefactID], CategoryArtefact):
                catArt = self.members[artefactID]
                catArt.getArtefact(includeMetadata=metadataRequired)
            else:
                raise Exception("CATEGORY LOADALLARTEFACTS ERROR: Member with ID '{}' is not a CategoryArtefact.".format(artefactID))
        
        return True
    
    def __str__(self):
        return """<Category instance
ID: {}
Name: {}
Description: {}
Members:{}
Created: {} />    
        """.format(
            self.id,
            self.name,
            self.description,
            ("\n---\n- " + ("\n- ".join(str(member) if isinstance(member, CategoryArtefact) else "CORRUPTED MEMBER AT '{}'".format(member) for member in self.members.values())) + "\n---") if isinstance(self.members, dict) else " None",
            self.created
        )
    
    @staticmethod
    def rawLoad(data: dict, categoryID: str | None=None, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Category':
        """Loads a category from raw data.

        Args:
            data (dict): The raw data to load the category from.
            categoryID (str | None, optional): The ID of the category. Defaults to None.
            withArtefacts (bool, optional): Whether to include artefacts in the loading process. Defaults to False.
            metadataRequired (bool, optional): Whether to include metadata in the Artefact loading process. Defaults to False (memory constraints).

        Returns:
            Category: The loaded category.
        """
        
        reqParam = ['name', 'description', 'members', 'created']
        for param in reqParam:
            if param not in data:
                if param == 'members':
                    data[param] = {}
                else:
                    data[param] = None
        
        cat = Category(
            name=data.get('name'),
            description=data.get('description'),
            members=None,
            created=data.get('created'),
            id=categoryID
        )
        
        members = {}
        if isinstance(data['members'], dict):
            for artefactID in data['members']:
                if isinstance(data['members'][artefactID], dict):
                    members[artefactID] = CategoryArtefact.rawLoad(
                        category=cat.id,
                        artefact=artefactID,
                        data=data['members'][artefactID],
                        withArtefact=withArtefacts,
                        metadataRequired=metadataRequired
                    )
        
        cat.members = members
        return cat
    
    @staticmethod
    def load(id: str=None, name: str=None, withMemberID: str=None, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Category | List[Category] | None':
        """Loads a category from the data source.

        Args:
            id (str, optional): Retrieve by ID of the category. Defaults to None.
            name (str, optional): Retrieve by the name of the category. Defaults to None.
            withMemberID (str, optional): Retrieve by a category that has a certain member. Defaults to None.
            withArtefacts (bool, optional): Whether to include artefacts in the loading process. Defaults to False.
            metadataRequired (bool, optional): Whether to include metadata in the Artefact loading process. Defaults to False (memory constraints).

        Raises:
            Exception: If an error occurs during loading.
            Exception: If a DIError occurs during loading.
            Exception: If the data format is unexpected.

        Returns:
            Category | List[Category] | None: The loaded category or categories.
        """
        
        if id != None:
            data = DI.load(Category.ref(id))
            if data is None:
                if name is not None or withMemberID is not None:
                    return Category.load(name=name, withMemberID=withMemberID, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("CATEGORY LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("CATEGORY LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Category.rawLoad(data, id, withArtefacts=withArtefacts, metadataRequired=metadataRequired)
        else:
            data = DI.load(Ref("categories"))
            if data is None:
                return None
            if isinstance(data, DIError):
                raise Exception("CATEGORY LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("CATEGORY LOAD ERROR: Failed to load dictionary categories data; response: {}".format(data))

            categories: Dict[str, Category] = {}
            for id in data:
                if isinstance(data[id], dict):
                    categories[id] = Category.rawLoad(data[id], id, withArtefacts=withArtefacts, metadataRequired=metadataRequired)

            if name is None and withMemberID is None:
                return list(categories.values())

            for category in categories.values():
                if name is not None and category.name == name:
                    return category
                elif withMemberID is not None and withMemberID in category.members:
                    return category
            
            return None
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("categories", id)

class CategoryArtefact(DIRepresentable):
    """
    Represents the relationship between a Category and an Artefact, including metadata about the association.
    This can be considered similar to a join table in relational databases, i.e a *..* relationship between Category and Artefact.

    Attributes:
        categoryID (str): ID of the category this artefact belongs to.
        artefactID (str): ID of the artefact.
        reason (str): Reason for including the artefact in the category.
        added (str): ISO timestamp when the artefact was added to the category.
        artefact (Artefact | None): The loaded Artefact object, if available.
        originRef (Ref): Reference object for database operations.

    Methods:
        getArtefact(): Loads the Artefact object for this CategoryArtefact.
        represent(): Returns a dictionary representation for serialization.
        save(): Saves the CategoryArtefact to the database.
        rawLoad(): Loads a CategoryArtefact from raw dictionary data.
        load(): Loads a CategoryArtefact by category and artefact.
        ref(): Returns the database reference for the CategoryArtefact.

    Sample usage:
    ```python
    from models import DI, Category, CategoryArtefact, Artefact
    DI.setup()

    # Load a category and its artefact relationship
    cat = Category.load(name="Paintings")
    catArt = cat.get("artefact123") # catArt is of type CategoryArtefact
    print(catArt.represent())

    # Load the full artefact object
    catArt.getArtefact(includeMetadata=True) # loads the Artefact object, passing includeMetadata=True to the Artefact.load() method
    print(catArt.artefact.represent())
    ```

    Notes:
        - The `reason` attribute should describe why the artefact is included in the category.
        - Use `getArtefact()` to load the full Artefact object if needed.
        - Changes to CategoryArtefact objects should be saved using `save()`.
    """
    
    def __init__(self, category: Category | str, artefact: Artefact | str, reason: str, added: str=None):
        """Initializes a CategoryArtefact instance.

        Args:
            category (Category | str): The category this artefact belongs to.
            artefact (Artefact | str): The artefact object or its ID.
            reason (str): The reason for including the artefact in the category.
            added (str, optional): ISO timestamp when the artefact was added to the category. Defaults to UTC now timestamp.
        """
        
        if added is None:
            added = Universal.utcNowString()
        
        self.categoryID = category.id if isinstance(category, Category) else category
        self.artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        self.reason = reason
        self.added = added
        self.artefact: Artefact = artefact if isinstance(artefact, Artefact) else None
        self.originRef = CategoryArtefact.ref(self.categoryID, self.artefactID)
    
    def getArtefact(self, includeMetadata: bool=False, safeFail: bool=False) -> bool:
        """Loads the Artefact object for this CategoryArtefact.

        Args:
            includeMetadata (bool, optional): Whether to include metadata when loading the Artefact. Defaults to False.
            safeFail (bool, optional): Whether to fail silently if the Artefact cannot be loaded. Defaults to False.

        Raises:
            Exception: If the artefactID is not a string.
            Exception: If the Artefact cannot be loaded and safeFail is False.

        Returns:
            bool: True if the Artefact was loaded successfully, False otherwise (False is returned only if safeFail=True).
        """
        
        if not isinstance(self.artefactID, str):
            raise Exception("CATEGORYARTEFACT GETARTEFACT ERROR: Invalid value set for 'artefactID'. Expected a string representing the Artefact ID.")
        
        self.artefact = Artefact.load(self.artefactID, includeMetadata=includeMetadata)
        if not isinstance(self.artefact, Artefact):
            if safeFail:
                self.artefact = None
                return False
            raise Exception("CATEGORYARTEFACT GETARTEFACT ERROR: Failed to load Artefact with ID '{}'; response: {}".format(self.artefactID, self.artefact))
        
        return True
    
    def represent(self) -> dict:
        """Represents the CategoryArtefact instance as a dictionary.

        Returns:
            dict: The dictionary representation of the CategoryArtefact.
        """
        
        return {
            "reason": self.reason,
            "added": self.added
        }
    
    def save(self):
        """Saves the CategoryArtefact instance.

        Returns:
            bool: True if the save operation was successful.
        """
        
        return DI.save(self.represent(), self.originRef)
    
    def __str__(self):
        return "CategoryArtefact(categoryID='{}', artefactID='{}', reason='{}', added='{}', artefactObject={})".format(
            self.categoryID,
            self.artefactID,
            self.reason,
            self.added,
            self.artefact if isinstance(self.artefact, Artefact) else "None"
        )
    
    @staticmethod
    def rawLoad(category: Category | str, artefact: Artefact | str, data: dict, withArtefact: bool=False, metadataRequired: bool=False) -> 'CategoryArtefact':
        """Loads a CategoryArtefact instance from raw data.

        Args:
            category (Category | str): The category this artefact belongs to.
            artefact (Artefact | str): The artefact object or its ID.
            data (dict): The raw data to load from.
            withArtefact (bool, optional): Whether to load the Artefact object when loading. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required when loading the Artefact. Defaults to False.

        Raises:
            Exception: If the categoryID is not a string.
            Exception: If the artefactID is not a string.

        Returns:
            CategoryArtefact: The loaded CategoryArtefact instance.
        """
        
        reqParams = ['reason', 'added']
        for reqParam in reqParams:
            if reqParam not in data:
                data[reqParam] = None
        
        if not isinstance(category, Category) and not isinstance(category, str):
            raise Exception("CATEGORY RAWLOAD ERROR: 'category' must be a Category object or a string representing the category ID.")
        if not isinstance(artefact, Artefact) and not isinstance(artefact, str):
            raise Exception("CATEGORY RAWLOAD ERROR: 'artefact' must be an Artefact object or a string representing the artefact ID.")
        
        catArt = CategoryArtefact(
            category=category,
            artefact=artefact,
            reason=data.get('reason'),
            added=data.get('added')
        )
        
        if withArtefact:
            catArt.getArtefact(includeMetadata=metadataRequired)
        
        return catArt
    
    @staticmethod
    def load(category: Category | str, artefact: Artefact | str, withArtefact: bool=False, metadataRequired: bool=False) -> 'CategoryArtefact | None':
        """Loads a CategoryArtefact instance.

        Args:
            category (Category | str): The category this artefact belongs to (object or ID).
            artefact (Artefact | str): The artefact object or its ID.
            withArtefact (bool, optional): Whether to load the Artefact object when loading. Defaults to False.
            metadataRequired (bool, optional): Whether metadata is required when loading the Artefact. Defaults to False.

        Raises:
            Exception: If the categoryID is not a string.
            Exception: If the artefactID is not a string.

        Returns:
            CategoryArtefact | None: The loaded CategoryArtefact instance.
        """
        
        categoryID = category.id if isinstance(category, Category) else category
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        data = DI.load(CategoryArtefact.ref(categoryID, artefactID))
        if data == None:
            return None
        if isinstance(data, DIError):
            raise Exception("CATEGORYARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("CATEGORYARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return CategoryArtefact.rawLoad(categoryID, artefact, data, withArtefact=withArtefact, metadataRequired=metadataRequired)
    
    @staticmethod
    def ref(categoryID: str, artefactID: str) -> Ref:
        return Ref("categories", categoryID, "members", artefactID)

class Face(DIRepresentable):
    embeddingsData: 'Dict[str, Dict[str, torch.Tensor]] | None' = None
    embeddingsFile: 'File | None' = File("embeddings.pth", "people")
    
    def __init__(self, figure: 'Figure | str', embeddings: 'List[FaceEmbedding] | Dict[str, FaceEmbedding]', embedsLoaded: bool=False):
        # Validate embedsLoaded
        if not isinstance(embedsLoaded, bool):
            raise Exception("FACE INIT ERROR: 'embedsLoaded' must be a boolean value.")
        
        # Extract figureID and validate embeddings
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE INIT ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if isinstance(embeddings, list):
            if not all(isinstance(embed, FaceEmbedding) for embed in embeddings):
                raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
            
            embeddings = {Universal.generateUniqueID(customLength=6): embedding for embedding in embeddings}
        elif isinstance(embeddings, dict):
            for embed in embeddings.values():
                if not isinstance(embed, FaceEmbedding):
                    raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
        else:
            raise Exception("FACE INIT ERROR: 'embeddings' must be a list or dictionary of FaceEmbedding objects.")
        
        # Instantiate Face object
        self.embeddings = embeddings
        self.embedsLoaded = embedsLoaded
        self.figureID = figureID
        self.originRef = Face.ref(figureID)
    
    def represent(self):
        return {
            "embeddings": {embeddingID: embedding.representForDB() for embeddingID, embedding in self.embeddings.items()}
        }
    
    def save(self, embeds=True):
        DI.save(self.represent(), self.originRef)
        
        if not self.embedsLoaded and embeds:
            Logger.log("FACE SAVE WARNING: Embeddings not loaded before saving, saveEmbeds call will be skipped. Figure ID: {}".format(self.figureID))
        elif embeds:
            self.saveEmbeds()
        
        return True
    
    def addEmbedding(self, embedding: FaceEmbedding, autoSave: bool=False, embedID: str=None) -> bool:
        if not isinstance(embedding, FaceEmbedding):
            raise Exception("FACE ADDEMBEDDING ERROR: 'embedding' must be a FaceEmbedding object.")
        if not self.embedsLoaded:
            raise Exception("FACE ADDEMBEDDING ERROR: Embeddings are not loaded; call loadEmbeds() first.")
        
        if embedID is None:
            embedID = Universal.generateUniqueID(customLength=6, notIn=list(self.embeddings.keys()))
        
        if embedID in self.embeddings:
            raise Exception("FACE ADDEMBEDDING ERROR: Embedding with ID '{}' already exists.".format(embedID))
        
        self.embeddings[embedID] = embedding
        
        if autoSave:
            self.saveEmbeds()
        
        return True
    
    def loadEmbeds(self, matchDBEmbeddings: bool=True) -> bool:
        if Face.embeddingsData is None:
            Face.loadEmbeddings()
        
        faceEmbeds = Face.extractEmbeddings(self.figureID) or {}
        
        changes = False
        if matchDBEmbeddings:
            # Remove embeddings that are not in the database
            for embeddingID in list(faceEmbeds.keys()):
                if embeddingID not in self.embeddings:
                    del faceEmbeds[embeddingID]
                    changes = True
        
        for embeddingID, embed in self.embeddings.items():
            if isinstance(embed, FaceEmbedding):
                embed.value = faceEmbeds.get(embeddingID, None)
        
        self.embedsLoaded = True
        if changes:
            self.saveEmbeds()
        
        return True
    
    def saveEmbeds(self):
        if not self.embedsLoaded:
            raise Exception("FACE SAVEEMBEDS ERROR: Embeddings are not loaded; call loadEmbeds() first.")
        
        if Face.embeddingsData is None:
            Face.loadEmbeddings()
        
        Face.setEmbeddings(self.figureID, {id: embed.value for id, embed in self.embeddings.items()})
        Face.saveEmbeddings()
        
        return True
    
    def offloadEmbeds(self):
        for embed in self.embeddings.values():
            embed.value = None
        
        self.embedsLoaded = False
        return True
    
    def destroy(self, embeds: bool=True):
        DI.save(None, self.originRef)
        
        if embeds:
            if Face.embeddingsData is None:
                Face.loadEmbeddings()
            
            Face.deleteFigure(self.figureID)
            Face.saveEmbeddings()
        
        return True
    
    def destroyEmbedding(self, embedID: str, includingValue: bool=True):
        if embedID in self.embeddings:
            del self.embeddings[embedID]
        
        if includingValue:
            self.saveEmbeds()
        
        return True
    
    def reload(self, withEmbeddings: bool=True, matchDBEmbeddings: bool=True):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("FACE RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("FACE RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("FACE RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.figureID, data, withEmbeddings, matchDBEmbeddings).__dict__)
        return True
    
    def __str__(self):
        return "Face(figureID='{}', embeddings={})".format(
            self.figureID,
            {embedID: str(embed) for embedID, embed in self.embeddings.items()} if isinstance(self.embeddings, dict) else None
        )
    
    @staticmethod
    def rawLoad(figureID: str, data: dict, withEmbeddings: bool=True, matchDBEmbeddings: bool=True) -> 'Face':
        dbEmbeds: Dict[str, Dict[str, str]] = data.get('embeddings', {})
        
        objectEmbeddings: Dict[str, FaceEmbedding] = {}
        for embeddingID, data in dbEmbeds.items():
            if isinstance(data, dict):
                objectEmbeddings[embeddingID] = FaceEmbedding(
                    value=None,
                    origin=data.get('origin', 'Unknown'),
                    added=data.get('added')
                )
        
        f = Face(figureID, objectEmbeddings)
        
        if withEmbeddings:
            f.loadEmbeds(matchDBEmbeddings)
        
        return f

    @staticmethod
    def load(figure: 'Figure | str', withEmbeddings: bool=True, matchDBEmbeddings: bool=True) -> 'Face | None':
        figureID = figure.id if isinstance(figure, Figure) else figure
        
        data = DI.load(Face.ref(figureID))
        if data == None:
            if matchDBEmbeddings:
                # If the figureID is not found, check if the embeddings file exists and delete the figure from it
                if Face.embeddingsData is None:
                    Face.loadEmbeddings()
                
                if figureID in Face.embeddingsData:
                    Face.deleteFigure(figureID)
                    Face.saveEmbeddings()
            
            return None
        if isinstance(data, DIError):
            raise Exception("FACE LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("FACE LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return Face.rawLoad(figureID, data, withEmbeddings, matchDBEmbeddings)
    
    @staticmethod
    def ensureEmbeddingsFile():
        fileExists = Face.embeddingsFile.exists()
        
        if isinstance(fileExists, str):
            raise Exception("FACE ENSUREEMBEDDINGSFILE ERROR: Failed to check if embeddings file exists; response: {}".format(fileExists))
        elif not fileExists[0]:
            Face.embeddingsData = {}
            Face.saveEmbeddings()
        elif not fileExists[1] or not FileManager.existsInSystem(Face.embeddingsFile):
            prep = FileManager.prepFile(file=Face.embeddingsFile)
            if not isinstance(prep, File):
                raise Exception("FACE ENSUREEMBEDDINGSFILE ERROR: Failed to prepare embeddings file; response: {}".format(prep))
        
        return True
    
    @staticmethod
    def loadEmbeddings():
        Face.ensureEmbeddingsFile()
        
        try:
            Face.embeddingsData = torch.load(Face.embeddingsFile.path(), map_location=Universal.device)
        except Exception as e:
            raise Exception("FACE LOADEMBEDDINGS ERROR: Failed to load embeddings data; error: {}".format(e)) from e
        
        return True
    
    @staticmethod
    def offloadEmbeddings(includingFile: bool=False):
        if Face.embeddingsData != None:
            Face.embeddingsData = None
        
        if includingFile and FileOps.exists(Face.embeddingsFile.path(), "file"):
            res = FileManager.offload(file=Face.embeddingsFile)
            if isinstance(res, str):
                raise Exception("FACE OFFLOADEMBEDDINGS ERROR: Failed to offload embeddings file; response: {}".format(res))
        
        return True
    
    @staticmethod
    def saveEmbeddings():
        try:
            torch.save(Face.embeddingsData, Face.embeddingsFile.path())
            
            saveResult = FileManager.save(file=Face.embeddingsFile)
            if isinstance(saveResult, str):
                raise Exception(saveResult)
        except Exception as e:
            raise Exception("FACE SAVEEMBEDDINGS ERROR: Failed to save embeddings data; error: {}".format(e)) from e

        return True
    
    @staticmethod
    def extractEmbeddings(figure: 'Figure | str'):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE EXTRACTEMBEDDINGS ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE EXTRACTEMBEDDINGS ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        if not isinstance(Face.embeddingsData, dict):
            return None
        
        return Face.embeddingsData.get(figureID, None)
    
    @staticmethod
    def extractEmbedding(figure: 'Figure | str', embeddingID: str):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE EXTRACTEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE EXTRACTEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE EXTRACTEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        return Face.embeddingsData.get(figureID, {}).get(embeddingID, None)
    
    @staticmethod
    def setEmbeddings(figure: 'Figure | str', embeddings: Dict[str, torch.Tensor]):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE SETEMBEDDINGS ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddings, dict):
            raise Exception("FACE SETEMBEDDINGS ERROR: 'embeddings' must be a dictionary of embeddings.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE SETEMBEDDINGS ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        Face.embeddingsData[figureID] = embeddings
        
        return True
    
    @staticmethod
    def setEmbedding(figure: 'Figure | str', embeddingID: str, embedding: torch.Tensor):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE SETEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE SETEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        if not isinstance(embedding, torch.Tensor):
            raise Exception("FACE SETEMBEDDING ERROR: 'embedding' must be a torch.Tensor object.")
        if Face.embeddingsData is None:
            raise Exception("FACE SETEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID not in Face.embeddingsData:
            Face.embeddingsData[figureID] = {}
        
        Face.embeddingsData[figureID][embeddingID] = embedding
        
        return True
    
    @staticmethod
    def deleteFigure(figure: 'Figure | str'):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE DELETEFIGURE ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE DELETEFIGURE ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID in Face.embeddingsData:
            del Face.embeddingsData[figureID]
        
        return True
    
    @staticmethod
    def deleteEmbedding(figure: 'Figure | str', embeddingID: str):
        figureID = figure.id if isinstance(figure, Figure) else figure
        if not isinstance(figureID, str):
            raise Exception("FACE DELETEEMBEDDING ERROR: 'figure' must be a Figure object or a string representing the figure ID.")
        if not isinstance(embeddingID, str):
            raise Exception("FACE DELETEEMBEDDING ERROR: 'embeddingID' must be a string representing the embedding ID.")
        
        if Face.embeddingsData is None:
            raise Exception("FACE DELETEEMBEDDING ERROR: Embeddings data is not loaded; call Face.loadEmbeddings() first.")
        
        if figureID in Face.embeddingsData and embeddingID in Face.embeddingsData[figureID]:
            del Face.embeddingsData[figureID][embeddingID]
        
        return True
    
    @staticmethod
    def ref(figureID: str) -> Ref:
        return Ref("figures", figureID, "face")

class Figure(DIRepresentable):
    def __init__(self, label: str, headshot: str, face: Face, profile: None=None, id: str=None):
        if id is None:
            id = Universal.generateUniqueID()
        if face is None:
            face = Face(id, [], embedsLoaded=True)
        elif not isinstance(face, Face):
            raise Exception("FIGURE INIT ERROR: 'face' must be a Face object.")
        
        self.id = id
        self.label = label
        self.headshot = headshot
        self.face = face
        self.profile = profile
        self.originRef = Figure.ref(id)
    
    def represent(self) -> dict:
        return {
            "label": self.label,
            "headshot": self.headshot,
            "face": self.face.represent() if isinstance(self.face, Face) else None,
            "profile": self.profile
        }
    
    def save(self, embeds: bool=True):
        DI.save(self.represent(), self.originRef)
        
        if not self.face.embedsLoaded and embeds:
            Logger.log("FIGURE SAVE WARNING: Embeddings not loaded before saving, saveEmbeds call will be skipped. Figure ID: {}".format(self.id))
        elif embeds:
            self.face.saveEmbeds()
        
        return True
    
    def destroy(self, embeds: bool=True):
        DI.save(None, self.originRef)
        
        if embeds:
            if Face.embeddingsData is None:
                Face.loadEmbeddings()
            
            Face.deleteFigure(self.id)
            Face.saveEmbeddings()
        
        return True
    
    def reload(self, withEmbeddings: bool=True, matchDBEmbeddings: bool=True):
        data = DI.load(self.originRef)
        if isinstance(data, DIError):
            raise Exception("FIGURE RELOAD ERROR: DIError occurred: {}".format(data))
        if data == None:
            raise Exception("FIGURE RELOAD ERROR: No data found at reference '{}'.".format(self.originRef))
        if not isinstance(data, dict):
            raise Exception("FIGURE RELOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        self.__dict__.update(self.rawLoad(self.id, data, withEmbeddings, matchDBEmbeddings).__dict__)
        return True
    
    @staticmethod
    def rawLoad(figureID: str, data: dict, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure':
        reqParams = ['label', 'headshot', 'face', 'profile']
        for reqParam in reqParams:
            if reqParam not in data:
                if reqParam == 'face':
                    data[reqParam] = {}
                else:
                    data[reqParam] = None
        
        face = None
        if isinstance(data['face'], dict):
            face = Face.rawLoad(figureID, data['face'], withEmbeddings, matchDBEmbeddings)
        
        return Figure(
            label=data['label'],
            headshot=data['headshot'],
            face=face,
            profile=data['profile'],
            id=figureID
        )
    
    @staticmethod
    def load(id: str=None, label: str=None, withEmbeddings: bool=False, matchDBEmbeddings: bool=True) -> 'Figure | List[Figure] | None':
        if id != None:
            data = DI.load(Figure.ref(id))
            if data is None:
                if label is not None:
                    return Figure.load(label=label, withEmbeddings=withEmbeddings, matchDBEmbeddings=matchDBEmbeddings)
                else:
                    return None
            if isinstance(data, DIError):
                raise Exception("FIGURE LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("FIGURE LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            return Figure.rawLoad(id, data, withEmbeddings, matchDBEmbeddings)
        else:
            data = DI.load(Ref("figures"))
            if data is None:
                return [] if label is None else None
            if isinstance(data, DIError):
                raise Exception("FIGURE LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("FIGURE LOAD ERROR: Failed to load dictionary figures data; response: {}".format(data))

            figures: Dict[str, Figure] = {}
            for id in data:
                if isinstance(data[id], dict):
                    figures[id] = Figure.rawLoad(id, data[id])

            if label is None:
                if withEmbeddings:
                    for figure in figures.values():
                        figure.face.loadEmbeds(matchDBEmbeddings)
                
                return list(figures.values())
            
            for figure in figures.values():
                if label is not None and figure.label == label:
                    if withEmbeddings:
                        figure.face.loadEmbeds(matchDBEmbeddings)
                    
                    return figure
            
            return None
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("figures", id)