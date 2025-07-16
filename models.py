from enum import Enum
from database import DI, DIRepresentable, DIError
from fm import File, FileManager
from utils import Ref
from services import Universal
from typing import List, Dict, Any, Literal

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
    def load(artefactID: str) -> 'MMData | None' :
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
    class Stage(str, Enum):
        UNPROCESSED = "unprocessed"
        PROCESSED = "processed"
        CONFIRMED = "confirmed"
        
        @staticmethod
        def validateAndReturn(stage) -> 'Batch.Stage | Literal[False]':
            if isinstance(stage, Batch.Stage):
                return stage
            elif isinstance(stage, str):
                try:
                    return Batch.Stage(stage.lower())
                except ValueError:
                    return False
            else:
                return False
    
    def __init__(self, userID: str, batchArtefacts: 'Dict[str, BatchArtefact]'=None, processingJob: str | None=None, cancelled: bool = False, created: str=None, id: str=None):
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if batchArtefacts is None:
            batchArtefacts = {}
        
        self.id: str = id
        self.userID: str = userID
        self.artefacts: Dict[str, BatchArtefact] = batchArtefacts
        self.processingJob: str | None = processingJob
        self.cancelled: bool = cancelled
        self.user: User | None = None
        self.created: str = created
        self.originRef = Batch.ref(self.id)

    def save(self, checkIntegrity: bool=True) -> bool:
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
                if batchArt.stage not in [Batch.Stage.UNPROCESSED, Batch.Stage.PROCESSED, Batch.Stage.CONFIRMED]:
                    raise Exception("BATCH SAVE ERROR: Invalid stage '{}' for artefact with ID '{}'.".format(batchArt.stage, artefactID))
        
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "userID": self.userID,
            "artefacts": {id: batchArt.represent() for id, batchArt in self.artefacts.items()},
            'processingJob': self.processingJob,
            'cancelled': self.cancelled,
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
        
        self.__dict__.update(self.rawLoad(data, self.id).__dict__)
        return True
    
    def __str__(self):
        completeData = self.represent()
        completeData['user'] = self.user.represent() if self.user is not None else None
        
        return str(completeData)
    
    def getUser(self) -> bool:
        if not isinstance(self.userID, str):
            raise Exception("BATCH GETUSER ERROR: Invalid value set for 'userID'. Expected a string representing the User ID.")
        
        self.user = User.load(id=self.userID)
        if not isinstance(self.user, User):
            raise Exception("BATCH GETUSER ERROR: Failed to load User with ID '{}'; response: {}".format(self.userID, self.user))
        
        return True
    
    def add(self, artefact: Artefact | str, to: 'Batch.Stage') -> 'BatchArtefact':
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
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return artefactID in self.artefacts
    
    def move(self, artefact: Artefact | str, to: 'Batch.Stage') -> bool:
        to = Batch.Stage.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH MOVE ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if not self.has(artefactID):
            raise Exception("BATCH MOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        batchArt = self.artefacts[artefactID]
        if batchArt.stage == to:
            raise Exception("BATCH MOVE ERROR: Artefact with ID '{}' is already in stage '{}'.".format(artefactID, to))
        batchArt.stage = to
        
        return True
    
    def where(self, artefact: Artefact | str) -> 'Batch.Stage | None':
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        return self.artefacts[artefactID].stage if self.has(artefactID) else None
    
    def remove(self, artefact: Artefact | str) -> bool:
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if not self.has(artefactID):
            raise Exception("BATCH REMOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        self.artefacts.pop(artefactID, None)
        return True
    
    def cancel(self, autoSave: bool=True):
        self.cancelled = True
        if autoSave:
            self.save(checkIntegrity=False)

    @staticmethod
    def rawLoad(data: dict, batchID: str | None=None) -> 'Batch':
        requiredParams = ['userID', 'unprocessed', 'processed', 'confirmed', 'processingJob', 'cancelled', 'created']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam in ['unprocessed', 'processed', 'confirmed']:
                    data[reqParam] = []
                elif reqParam in ['cancelled']:
                    data[reqParam] = False
                else:
                    data[reqParam] = None
        
        if not isinstance(data['unprocessed'], list):
            data['unprocessed'] = []
        if not isinstance(data['processed'], list):
            data['processed'] = []
        if not isinstance(data['confirmed'], list):
            data['confirmed'] = []
        
        return Batch(
            userID=data['userID'],
            unprocessed=data['unprocessed'],
            processed=data['processed'],
            confirmed=data['confirmed'],
            processingJob=data['processingJob'],
            cancelled=data['cancelled'],
            created=data['created'],
            id=batchID
        )
    
    @staticmethod
    def load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch | Dict[Batch] | None':
        """
        Loads batches from the data store.
        
        Args:
            id: Specific batch ID to load
            userID: User ID to filter by
            withUser: Whether to load user objects
            withArtefacts: Whether to load artefact objects
            metadataRequired: Whether to load artefact metadata

        Returns:
            Matching Batch(es) or None if not found

        Raises:
            Exception: If data loading fails
        """
        if id != None:
            data = DI.load(Batch.ref(id))
            if data is None:
                return None
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            batch = Batch.rawLoad(data, id)
            
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
                    batches[id] = Batch.rawLoad(data[id], id)
            
            if userID is None:
                for batch in batches.values():
                    if withUser:
                        batch.getUser()
                    if withArtefacts:
                        batch.getArtefacts(metadataRequired=metadataRequired)
                
                return list(batches.values())
            
            userBatches: List[Batch] = []
            for batchID in batches:
                if batches[batchID].userID == userID:
                    userBatches.append(batches[batchID])
            
            for batch in userBatches:
                if withUser:
                    batch.getUser()
                if withArtefacts:
                    batch.getArtefacts(metadataRequired=metadataRequired)
            
            return userBatches
    
    @staticmethod
    def ref(id: str) -> Ref:
        return Ref("batches", id)

class BatchArtefact(DIRepresentable):
    def __init__(self, batch: Batch | str, artefact: Artefact | str, stage: Batch.Stage, processedDuration: str | None, processedTime: str | None=None, processingError: str | None=None):
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
    
    @staticmethod
    def rawLoad(data: dict, batchID: str | None=None, artefactID: str | None=None):
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
        
        return BatchArtefact(
            batch=batchID,
            artefact=artefactID,
            stage=stage,
            processedDuration=data.get('processedDuration'),
            processedTime=data.get('processedTime'),
            processingError=data.get('processingError')
        )
    
    @staticmethod
    def load(batch: Batch | str, artefact: Artefact | str) -> 'BatchArtefact | None':
        batchID = batch.id if isinstance(batch, Batch) else batch
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        data = DI.load(BatchArtefact.ref(batchID, artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("BATCHARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("BATCHARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
        
        return BatchArtefact.rawLoad(data, batchID, artefactID)
    
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
    
    def add(self, artefact: Artefact | str, reason: str, checkIntegrity: bool=True, loadArtefactObject: bool=False, metadataRequired: bool=False) -> bool:
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
        
        if checkIntegrity:
            artefact = Artefact.load(artefactID, includeMetadata=False)
            if not isinstance(artefact, Artefact):
                raise Exception("CATEGORY ADD ERROR: Failed to load Artefact with ID '{}'; response: {}".format(artefactID, artefact))
        
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