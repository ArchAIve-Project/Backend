from enum import Enum
from database import DI, DIRepresentable, DIError
from fm import File, FileManager
from utils import Ref
from services import Universal
from typing import List, Dict, Any, Literal
from datetime import datetime 

class Artefact(DIRepresentable):
    def __init__(self, name: str, image: str, metadata: 'Metadata | Dict[str, Any] | None', public: bool = False, created: str=None, metadataLoaded: bool=True, id: str = None):
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
    
    def getFMFile(self) -> File:
        return File(self.image, "artefacts")
    
    def fmSave(self):
        fmFile = self.getFMFile()
        return FileManager.save(fmFile.store, fmFile.filename)
    
    def fmDelete(self):
        fmFile = self.getFMFile()
        return FileManager.delete(file=fmFile)

    def represent(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'public': self.public,
            'name': self.name,
            'image': self.image,
            'created': self.created,
            'metadata': self.metadata.represent() if self.metadata is not None else None
        }

    @staticmethod
    def rawLoad(data, includeMetadata: bool=True):
        requiredParams = ['public', 'name', 'image', 'created', 'metadata', 'id']
        for param in requiredParams:
            if param not in data:
                if param == 'public':
                    data['public'] = False
                else:
                    data[param] = None

        metadata = None
        if includeMetadata and isinstance(data['metadata'], dict):
            metadata = Metadata.rawLoad(data['id'], data['metadata'])

        output = Artefact(
            name=data['name'], 
            image=data['image'],
            metadata=metadata,
            public=data['public'],
            created=data['created'],
            metadataLoaded=includeMetadata,
            id=data['id']
        )

        return output

    @staticmethod
    def load(id: str = None, name: str=None, image: str=None, includeMetadata: bool=True) -> 'List[Artefact] | Artefact | None':
        if id is not None:
            data = DI.load(Artefact.ref(id))
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if data is None:
                if name is not None or image is not None:
                    return Artefact.load(name=name, image=image)
                else:
                    return None
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return Artefact.rawLoad(data, includeMetadata)
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
                    artefacts[id] = Artefact.rawLoad(data[id], includeMetadata)
            
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
        # 1. identify whether it is mm data or hf data
        # 2. manually feed in the values to either constructor to product a MMData | HFData object
        # 3. Pass the data object to Metadata constructor
        # 4. Return the Metadata object
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
            "mmIDs": self.mmIDs,
            "id": self.id
        }
    
    @staticmethod
    def rawLoad(data: dict) -> 'Book':
        reqParams = ['title', 'subtitle', 'mmIDs', 'id']
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
            id=data.get('id')
        )
    
    @staticmethod
    def load(id: str=None, title: str=None, withMMID: str=None) -> 'Book | List[Book] | None':
        '''
        Load a book by its ID, title, or MMID.
        If id is provided, it loads the specific book.
        If title is provided, it searches for a book with that title.
        If withMMID is provided, it searches for a book that contains that MMID in its mmIDs list.
        If neither id nor title is provided, it loads all books and returns the first one that matches the criteria.
        '''
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

            return Book.rawLoad(data)

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
                    books[id] = Book.rawLoad(data[id])
            
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
    
    def __init__(self, userID: str, unprocessed: List[Artefact | str]=None, processed: List[Artefact | str]=None, confirmed: List[Artefact | str]=None, created: str=None, id: str=None):
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if unprocessed is None:
            unprocessed = []
        if processed is None:
            processed = []
        if confirmed is None:
            confirmed = []

        if not isinstance(unprocessed, list):
            raise TypeError("BATCH INIT ERROR: 'unprocessed' must be a list of Artefact or str.")
        if not isinstance(processed, list):
            raise TypeError("BATCH INIT ERROR: 'processed' must be a list of Artefact or str.")
        if not isinstance(confirmed, list):
            raise TypeError("BATCH INIT ERROR: 'confirmed' must be a list of Artefact or str.")
        
        unprocessedData = {}
        processedData = {}
        confirmedData = {}
        
        for artefactItem in unprocessed:
            if isinstance(artefactItem, Artefact):
                unprocessedData[artefactItem.id] = artefactItem
            elif isinstance(artefactItem, str):
                unprocessedData[artefactItem] = None
            else:
                raise TypeError("BATCH INIT ERROR: 'unprocessed' list must contain Artefact objects or strings representing Artefact IDs.")
        
        for artefactItem in processed:
            if isinstance(artefactItem, Artefact):
                processedData[artefactItem.id] = artefactItem
            elif isinstance(artefactItem, str):
                processedData[artefactItem] = None
            else:
                raise TypeError("BATCH INIT ERROR: 'processed' list must contain Artefact objects or strings representing Artefact IDs.")
        
        for artefactItem in confirmed:
            if isinstance(artefactItem, Artefact):
                confirmedData[artefactItem.id] = artefactItem
            elif isinstance(artefactItem, str):
                confirmedData[artefactItem] = None
            else:
                raise TypeError("BATCH INIT ERROR: 'confirmed' list must contain Artefact objects or strings representing Artefact IDs.")
        
        self.id: str = id
        self.userID: str = userID
        self.unprocessed: Dict[str, Artefact | None] = unprocessedData
        self.processed: Dict[str, Artefact | None] = processedData
        self.confirmed: Dict[str, Artefact | None] = confirmedData
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
            
            allIDs = []
            for stage in [self.unprocessed, self.processed, self.confirmed]:
                for id in stage:
                    if id in allIDs:
                        raise Exception("BATCH SAVE ERROR: Duplicate Artefact ID '{}' from stage '{}' found.".format(id, stage))
                    allIDs.append(id)
        
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "userID": self.userID,
            "unprocessed": list(self.unprocessed.keys()),
            "processed": list(self.processed.keys()),
            "confirmed": list(self.confirmed.keys()),
            "created": self.created
        }
    
    def __str__(self):
        completeData = self.represent()
        completeData['user'] = self.user.represent() if self.user is not None else None
        completeData['unprocessed'] = {id: (self.unprocessed[id].represent() if isinstance(self.unprocessed[id], Artefact) else None) for id in self.unprocessed.keys()}
        completeData['processed'] = {id: (self.processed[id].represent() if isinstance(self.processed[id], Artefact) else None) for id in self.processed.keys()}
        completeData['confirmed'] = {id: (self.confirmed[id].represent() if isinstance(self.confirmed[id], Artefact) else None) for id in self.confirmed.keys()}
        
        return str(completeData)
    
    def getUser(self) -> bool:
        if self.userID is None:
            raise Exception("BATCH GETUSER ERROR: Batch does not have a userID set.")
        
        self.user = User.load(id=self.userID)
        
        return True if isinstance(self.user, User) else False
    
    def getArtefacts(self, unprocessed: bool=True, processed: bool=True, confirmed: bool=True, metadataRequired: bool=False) -> bool:
        if not (unprocessed or processed or confirmed):
            raise Exception("BATCH GETARTEFACTS ERROR: At least one of unprocessed, processed, or confirmed must be True.")
        
        for stageDict in [self.unprocessed, self.processed, self.confirmed]:
            for id in stageDict:
                if stageDict[id] is None:
                    data = Artefact.load(id=id, includeMetadata=metadataRequired)
                    if not isinstance(data, Artefact):
                        raise Exception("BATCH GETARTEFACTS ERROR: Failed to load Artefact with ID '{}'; response: {}".format(id, data))
                    stageDict[id] = data
        
        return True
    
    def add(self, artefact: Artefact | str, to: 'Batch.Stage') -> bool:
        to = Batch.Stage.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH ADD ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        artefactObject = artefact if isinstance(artefact, Artefact) else None
        
        if self.has(artefactID):
            raise Exception("BATCH ADD ERROR: Artefact with ID '{}' already exists in the batch.".format(artefactID))
        
        if to == Batch.Stage.UNPROCESSED:
            self.unprocessed[artefactID] = artefactObject
        elif to == Batch.Stage.PROCESSED:
            self.processed[artefactID] = artefactObject
        elif to == Batch.Stage.CONFIRMED:
            self.confirmed[artefactID] = artefactObject
        
        return True
    
    def has(self, artefact: Artefact | str) -> bool: 
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        allIDs = list(self.unprocessed.keys()) + list(self.processed.keys()) + list(self.confirmed.keys())
        return artefactID in allIDs
    
    def move(self, artefact: Artefact | str, to: 'Batch.Stage') -> bool:
        to = Batch.Stage.validateAndReturn(to)
        if to == False:
            raise Exception("BATCH MOVE ERROR: Invalid stage provided; expected 'Batch.Stage' or string representation of it.")
        
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        artefactObject = artefact if isinstance(artefact, Artefact) else None
        
        if not self.has(artefactID):
            raise Exception("BATCH MOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        # Find which stage the artefact is currently in
        for stage in [self.unprocessed, self.processed, self.confirmed]:
            if artefactID in stage:
                del stage[artefactID]
                break
        
        # Add the artefact to the new stage
        if to == Batch.Stage.UNPROCESSED:
            self.unprocessed[artefactID] = artefactObject
        elif to == Batch.Stage.PROCESSED:
            self.processed[artefactID] = artefactObject
        elif to == Batch.Stage.CONFIRMED:
            self.confirmed[artefactID] = artefactObject
        
        return True
    
    def where(self, artefact: Artefact | str) -> 'Batch.Stage | None':
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if artefactID in self.unprocessed:
            return Batch.Stage.UNPROCESSED
        elif artefactID in self.processed:
            return Batch.Stage.PROCESSED
        elif artefactID in self.confirmed:
            return Batch.Stage.CONFIRMED
        else:
            return None
    
    def remove(self, artefact: Artefact | str) -> bool:
        artefactID = artefact.id if isinstance(artefact, Artefact) else artefact
        
        if not self.has(artefactID):
            raise Exception("BATCH REMOVE ERROR: Artefact with ID '{}' does not exist in the batch.".format(artefactID))
        
        # Find which stage the artefact is currently in and remove it
        stage = self.where(artefactID)
        if stage == Batch.Stage.UNPROCESSED:
            del self.unprocessed[artefactID]
        elif stage == Batch.Stage.PROCESSED:
            del self.processed[artefactID]
        elif stage == Batch.Stage.CONFIRMED:
            del self.confirmed[artefactID]
        
        return True

    @staticmethod
    def rawLoad(data: dict) -> 'Batch':
        requiredParams = ['userID', 'unprocessed', 'processed', 'confirmed', 'created', 'id']
        for reqParam in requiredParams:
            if reqParam not in data or data[reqParam] is None:
                if reqParam in ['unprocessed', 'processed', 'confirmed']:
                    data[reqParam] = []
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
            created=data['created'],
            id=data['id']
        )
    
    @staticmethod
    def load(id: str=None, userID: str=None, withUser: bool=False, withArtefacts: bool=False, metadataRequired: bool=False) -> 'Batch | Dict[Batch] | None':
        '''
        If id found load specific batch from the database
        else loads all batches from the database and filter based on userID // Gets all batch under that userID

        Args:
            id (str, optional): The ID of the batch to load. Defaults to None.
            userID (str, optional): The user ID of the batch to load. Defaults to None.
            withUser (bool, optional): Whether to include user information in the loaded batch. Defaults to False.
            withArtefacts (bool, optional): Whether to include artefacts in the loaded batch. Defaults to False.

        Returns:
            Batch | Dict[Batch] | None: The loaded batch or list of batches, or None if not found.
        '''
        if id != None:
            data = DI.load(Batch.ref(id))
            if data is None:
                return None
            if isinstance(data, DIError):
                raise Exception("BATCH LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("BATCH LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

            batch = Batch.rawLoad(data)
            
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
                    batches[id] = Batch.rawLoad(data[id])
            
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
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "pwd": self.pwd,
            "authToken": self.authToken,
            "superuser": self.superuser,
            "lastLogin": self.lastLogin,
            "created": self.created
        }
    
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
    
    @staticmethod
    def rawLoad(data) -> 'User':
        requiredParams = ['username', 'email', 'pwd', 'authToken', 'superuser', 'lastLogin', 'created', 'id']
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
            id=data['id']
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
            
            return User.rawLoad(data)
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
                    users[id] = User.rawLoad(data[id])

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

if __name__ == "__main__":
    # data = {
    #     "traditional_chinese": "傳統中文",
    #     "correction_applied": True,
    #     "pre_correction_accuracy": "Pre-correction accuracy",
    #     "post_correction_accuracy": "Post-correction accuracy",
    #     "simplified_chinese": "简体中文",
    #     "english_translation": "English translation",
    #     "summary": "Summary of the artefact",
    #     "ner_labels": "NER labels for the artefact"
    # }

    DI.setup()
    # art = Artefact("hello", "image.png", data)

    # print(art)

    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print("ERROR: {}".format(e))