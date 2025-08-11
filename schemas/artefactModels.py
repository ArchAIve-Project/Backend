from database import DI, DIRepresentable, DIError
from fm import File, FileManager
from utils import Ref
from services import Universal
from typing import List, Dict, Any

class Artefact(DIRepresentable):
    """
    Represents an artefact (e.g., image or document) within the DI system.

    Artefacts contain metadata and are associated with stored files. This class handles the creation,
    persistence, file management, and metadata binding of artefact objects.
    """
    def __init__(self, name: str, image: str, metadata: 'Metadata | Dict[str, Any] | None', description: str=None, public: bool = False, created: str=None, artType: str | None=None, metadataLoaded: bool=True, id: str = None):
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
        
        if description is None:
            if metadata is not None:
                if metadata.isMM():
                    summary = metadata.raw.summary
                    artDescription = summary[:30] + "..." if len(summary) > 30 else summary
                elif metadata.isHF():
                    caption = metadata.raw.caption
                    artDescription = caption[:30] + "..." if len(caption) > 30 else caption
                else:
                    artDescription = None
            else:
                artDescription = None
        else:
            artDescription = description

        self.id = id
        self.name = name
        self.image = image
        self.description = artDescription
        self.public = public
        self.created = created
        self.metadata: Metadata | None = metadata
        self.artType: str | None = artType
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
            'description': self.description,
            'created': self.created,
            'metadata': self.metadata.represent() if self.metadata is not None else None
        }
    
    def __str__(self):
        return "Artefact(id='{}', name='{}', image='{}', description='{}', public={}, created='{}', metadataLoaded={}, metadata={})".format(
            self.id,
            self.name,
            self.image,
            self.description,
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
        requiredParams = ['public', 'name', 'image', 'description', 'created', 'metadata']
        for param in requiredParams:
            if param not in data:
                if param == 'public':
                    data['public'] = False
                else:
                    data[param] = None

        metadata = None
        artType: str | None = None
        if isinstance(data['metadata'], dict) and isinstance(artefactID, str):
            if includeMetadata:
                metadata = Metadata.rawLoad(artefactID, data['metadata'])
            artType = 'mm' if data['metadata'].get('tradCN', None) != None else 'hf'

        output = Artefact(
            name=data['name'], 
            image=data['image'],
            metadata=metadata,
            description=data['description'],
            public=data['public'],
            created=data['created'],
            artType=artType,
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
        elif 'figureIDs' in data:
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
        elif 'figureIDs' in metagenDict:
            hf = HFData(
                artefactID=artefactID,
                figureIDs=metagenDict.get('figureIDs'),
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
    def __init__(self, artefactID: str, figureIDs: list[str], caption: str, addInfo: str=None):
        """
        Metadata for human figure(s) in artefacts.
    
        Contains information about human figures detected in images, including references
        to Figure objects, captions describing the scene (e.g., "people in a meeting"), and optional additional
        information. Used for artefacts containing human imagery that requires annotation.

        Args:
            artefactID (str): Associated artefact identifier.
            figureIDs (list[str]): List of IDs referring to Figure objects detected in the artefact.
            caption (str): Descriptive caption for the human figures.
            addInfo (str | None): Additional context about the figures if available.
        """
        self.artefactID = artefactID
        self.figureIDs = figureIDs
        self.caption = caption
        self.addInfo = addInfo
        self.originRef = HFData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "figureIDs": self.figureIDs,
            "caption": self.caption,
            "addInfo": self.addInfo
        }

    @staticmethod
    def rawLoad(data: Dict[str, Any], artefactID: str) -> 'HFData':
        requiredParams = ['figureIDs', 'caption', 'addInfo']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'figureIDs':
                    data['figureIDs'] = []
                else:
                    data[reqParam] = None

        return HFData(
            artefactID=artefactID,
            figureIDs=data.get('figureIDs', []),
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
