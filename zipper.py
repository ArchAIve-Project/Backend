from services import Logger
from firebase import FireConn
from fm import FileManager, FireStorage, File
from database import DI, Ref
from schemas.artefactModels import Artefact
from schemas.figureModels import Figure

Logger.setup()

res = FireConn.connect()
if res != True:
    raise Exception("FIREBASE INIT ERROR: " + str(res))

res = FileManager.setup()
if res != True:
    raise Exception("FILEMANAGER INIT ERROR: " + str(res))

DI.setup()

# arts: list[Figure] = Figure.load()
# for art in arts:
#     FileManager.prepFile(store="people", filename=art.headshot)