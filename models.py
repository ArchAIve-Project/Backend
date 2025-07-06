from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal
from typing import List, Dict, Any

class Artefact(DIRepresentable):
    def __init__(self, name: str, image: str, metadata: 'Metadata | Dict[str, Any] | None', public: bool = False, created: str=None, id: str = None):
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()
        if isinstance(metadata, dict):
            metadata = Metadata.fromMetagen(id, metadata)
        if not isinstance(metadata, Metadata):
            metadata = Metadata(artefactID=id)

        self.id = id
        self.name = name
        self.image = image
        self.public = public
        self.created = created
        self.metadata: Metadata = metadata
        self.originRef = Artefact.ref(self.id)
    
    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'public': self.public,
            'name': self.name,
            'image': self.image,
            'created': self.created,
            'metadata': self.metadata.represent() if self.metadata is not None else {}
        }

    @staticmethod
    def rawLoad(data):
        requiredParams = ['public', 'name', 'image', 'created', 'metadata', 'id']
        for param in requiredParams:
            if param not in data:
                if param == 'public':
                    data['public'] = False
                else:
                    data[param] = None

        metadata = None
        if isinstance(data['metadata'], dict):
            metadata = Metadata.rawLoad(data['id'], data['metadata'])

        output = Artefact(
            name=data['name'], 
            image=data['image'],
            metadata=metadata,
            public=data['public'],
            created=data['created'], 
            id=data['id']
        )

        return output

    @staticmethod
    def load(id: str = None) -> 'List[Artefact] | Artefact | None':
        if id is not None:
            data = DI.load(Artefact.ref(id))
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if data is None:
                return None
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return Artefact.rawLoad(data)
        else:
            data = DI.load(Ref("artefacts"))
            if data == None:
                return None
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Failed to load dictionary artefacts data; response: {}".format(data))

            artefacts: Dict[str, Artefact] = {}
            for id in data:
                if isinstance(data[id], dict):
                    artefacts[id] = Artefact.rawLoad(data[id])
            
            return list(artefacts.values())
        
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
    def __init__(self, artefactID: str, faceFiles: list[str], caption: str):
        self.artefactID = artefactID
        self.faceFiles = faceFiles
        self.caption = caption
        self.originRef = HFData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def represent(self) -> Dict[str, Any]:
        return {
            "faceFiles": self.faceFiles,
            "caption": self.caption
        }

    @staticmethod
    def rawLoad(data: Dict[str, Any], artefactID: str) -> 'HFData':
        requiredParams = ['faceFiles', 'caption']
        for reqParam in requiredParams:
            if reqParam not in data:
                if reqParam == 'faceFiles':
                    data['faceFiles'] = []
                else:
                    data[reqParam] = None

        return HFData(
            artefactID=artefactID,
            faceFiles=data['faceFiles'],
            caption=data['caption']
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

if __name__ == "__main__":
    data = {
        "traditional_chinese": "傳統中文",
        "correction_applied": True,
        "pre_correction_accuracy": "Pre-correction accuracy",
        "post_correction_accuracy": "Post-correction accuracy",
        "simplified_chinese": "简体中文",
        "english_translation": "English translation",
        "summary": "Summary of the artefact",
        "ner_labels": "NER labels for the artefact"
    }

    DI.setup()
    art = Artefact("hello", "image.png", data)

    print(art)

    while True:
        try:
            exec(input(">>> "))
        except Exception as e:
            print("ERROR: {}".format(e))