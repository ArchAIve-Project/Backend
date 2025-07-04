from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal
from typing import List

class Artefact(DIRepresentable):
    def __init__(self, name: str, image: str, public: bool = False, created: str=None, id: str = None):
        if id is None:
            id = Universal.generateUniqueID()
        if created is None:
            created = Universal.utcNowString()

        self.id = id
        self.name = name
        self.image = image
        self.public = public
        self.created = created
        self.originRef = Artefact.ref(self.id)
    
    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)
    
    def represent(self) -> dict:
        return {
            'id': self.id,
            'public': self.public,
            'name': self.name,
            'image': self.image,
            'created': self.created
        }

    @staticmethod
    def rawLoad(data):
        requiredParams = ['public', 'name', 'image', 'created', 'id']
        for param in requiredParams:
            if param not in data:
                if param == 'created':
                    data['created'] = Universal.utcNowString()
                elif param == 'id':
                    data['id'] = Universal.generateUniqueID()
                elif param == 'public': 
                    data['public'] = False 
                else:
                    data[param] = None

        return Artefact(
            public=data['public'],
            name=data['name'], 
            image=data['image'], 
            created=data['created'], 
            id=data['id']
        )

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

class MMData(DIRepresentable):
    def __init__(self, artefactID: str, image: str, tradCN: str, preCorrectionAcc: str, postCorrectionAcc : str, simplifiedCNN : str, english : str, summary: str, nerLabels: str, correctionApplies: bool = False):
        self.artefactID = artefactID
        self.image = image
        self.tradCN = tradCN
        self.correctionApplies = correctionApplies
        self.preCorrectionAcc = preCorrectionAcc
        self.postCorrectionAcc = postCorrectionAcc
        self.simplifiedCNN = simplifiedCNN
        self.english = english
        self.summary = summary
        self.nerLabels = nerLabels
        self.originRef = MMData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)

    def represent(self) -> dict[str, any]:
        return {
            "image": self.image,
            "tradCN": self.tradCN,
            "correctionApplies": self.correctionApplies,
            "preCorrectionAcc": self.preCorrectionAcc,
            "postCorrectionAcc": self.postCorrectionAcc,
            "simplifiedCNN": self.simplifiedCNN,
            "english": self.english,
            "summary": self.summary,
            "nerLabels": self.nerLabels
        }

    @staticmethod
    def rawLoad(data: dict[str, any], artefactID: str) -> 'MMData':
        requiredParams = ['image', 'tradCN', 'correctionApplies','preCorrectionAcc','simplifiedCNN','english','summary','nerLabels']
        for reqParam in requiredParams:
            if reqParam == 'public': 
                data['public'] = False 
            else:
                data[reqParam] = None

        return MMData(
            artefactID=artefactID,                
            image=data['image'],
            tradCN=data['tradCN'],
            correctionApplies=data['correctionApplies'] == 'True',
            preCorrectionAcc=data['preCorrectionAcc'],                
            postCorrectionAcc=data['postCorrectionAcc'],
            simplifiedCNN=data['simplifiedCNN'],
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
        return Ref("artefacts", artefactID, "mmData")
    
class HFData(DIRepresentable):
    def __init__(self, artefactID: str, faceFiles: list[str], caption: str):
        self.artefactID = artefactID
        self.faceFiles = faceFiles
        self.caption = caption
        self.originRef = HFData.ref(artefactID)

    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)

    def destroy(self):
        return DI.save(None, self.originRef)

    def represent(self) -> dict[str, any]:
        return {
            "faceFiles": self.faceFiles,
            "caption": self.caption
        }

    @staticmethod
    def rawLoad(data: dict[str, any], artefactID: str) -> 'HFData':
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
            caption=data['caption'])

    @staticmethod
    def load(artefactID: str) -> 'HFData | None':
        data = DI.load(HFData.ref(artefactID))
        if data is None:
            return None
        if isinstance(data, DIError):
            raise Exception("HFData LOAD ERROR: DIError occurred: {}".format(data))
        if not isinstance(data, dict):
            raise Exception("HFData LOAD ERROR: Unexpected DI load response format; response: {}".format(data))

        return HFData.rawLoad(data, artefactID)

    @staticmethod
    def ref(artefactID: str) -> Ref:
        return Ref("artefacts", artefactID, "hfData")

