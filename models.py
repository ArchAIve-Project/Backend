from database import DI, DIRepresentable, DIError
from utils import Ref
from services import Universal

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
        self.originRef = self.ref(self.id)

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
    def load(id: str = None) -> 'Artefact' :
        if id is None:
            raise Exception("ARTEFACT LOAD ERROR: No ID provided.")
        if id != None:
            data = DI.load(Artefact.ref("artefacts", id)) 
            if isinstance(data, DIError):
                raise Exception("ARTEFACT LOAD ERROR: DIError occurred: {}".format(data))
            if data is None:
                return None
            if not isinstance(data, dict):
                raise Exception("ARTEFACT LOAD ERROR: Unexpected DI load response format; response: {}".format(data))
            
            return Artefact.rawLoad(data)

    @staticmethod
    def ref(id:str) -> Ref:
        return Ref("artefact", id)
    
    def save(self) -> bool:
        return DI.save(self.represent(), self.originRef)
    
    def destroy(self):
        return DI.save(None, self.originRef)

if __name__ == "__main__":

    DI.setup()

    testArtefact = Artefact(
        name= 'test Artefact',
        image= 'test.jpg'  
    )

    print("Created Artefact:", testArtefact.represent())
    
    testArtefact.save()
    print("Saved Artefact with ID:", testArtefact.id)
    