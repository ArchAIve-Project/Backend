import os, json, shutil, time
from dotenv import load_dotenv
load_dotenv()
from database import DI, Ref
from firebase import FireConn
from fm import FileManager, File
from services import Encryption, FileOps, ThreadManager, Logger, LiteStore
from ai import LLMInterface
from addons import ModelStore, ArchSmith
from cnnclassifier import ImageClassifier
from ccrPipeline import CCRPipeline
from NERPipeline import NERPipeline
from captioner import ImageCaptioning, Vocabulary
from schemas import artefactModels, collectionModels, figureModels, userModels
from ingestion import DataImportProcessor
from catalogueIntegration import HFCatalogueIntegrator

# Phase: Manual system setup
Logger.setup()
LiteStore.setup()
LLMInterface.initDefaultClients()
ThreadManager.initDefault()
ArchSmith.setup()
ModelStore.setup(
    autoLoad=os.environ.get("LLM_INFERENCE", "False") != "True",
    ccr=CCRPipeline.loadChineseClassifier,
    ccrCharFilter=CCRPipeline.loadBinaryClassifier,
    cnn=ImageClassifier.load_model,
    imageCaptionerResNet=ImageCaptioning.loadModel(weights='resnet'),
    imageCaptionerViT=ImageCaptioning.loadModel(weights='vit')
)
NERPipeline.load_model()

res = FireConn.connect()
if res != True:
    raise Exception("FIREBASE INIT ERROR: " + str(res))

res = FileManager.setup()
if res != True:
    raise Exception("FILEMANAGER INIT ERROR: " + str(res))

# Save existing data to a cache file
cache = DI.load()
with open("prePTCache.json", "w") as f:
    json.dump(cache, f, indent=4)

print("PT: DB cache saved to 'prePTCache.json'.")

# Phase: Reset
# Clear non-figures data
DI.save(None, Ref("artefacts"))
DI.save(None, Ref("auditLogs"))
DI.save(None, Ref("batches"))
DI.save(None, Ref("books"))
DI.save(None, Ref("categories"))
DI.save(None, Ref("users"))

print("PT: Non-figure data cleared from database.")

# Delete FileStore and artefacts stores from FileManager
FileManager.deleteAll(stores=["FileStore", "artefacts"])

print("PT: FileManager stores 'FileStore' and 'artefacts' cleared.")

# Phase: Seed Batch and Admin User
user = userModels.User("adminuser", "admin@email.com", Encryption.encodeToSHA256("123456"), "ArchAIve", "Admin", "Platform Superuser", superuser=True)
user.save()

print("PT: Superuser created with username 'adminuser' and password '123456'.")

# Setup batch
print("PT: Uploading seed artefacts...")
artefacts: list[artefactModels.Artefact] = []
for filename in FileOps.getFilenames("ptSeed"):
    file = File(filename, "artefacts")
    
    # Save to local FM store first
    shutil.copy(os.path.join(os.getcwd(), "ptSeed", filename), file.path())
    
    # Upload to cloud FM store
    res = FileManager.save(file=file)
    if isinstance(res, str):
        print("PT WARNING: Failed to upload seed artefact file '{}' to cloud store; response: {}".format(filename, res))
        continue
    
    art = artefactModels.Artefact.fromFMFile(file)
    art.save()
    
    artefacts.append(art)

print("PT: Seed artefacts uploaded and artefact records created.")

batch = collectionModels.Batch(user.id, "Seed Batch", artefacts)
batch.stage = collectionModels.Batch.Stage.UNPROCESSED
batch.save()

print("PT: Seed batch '{}' created with {} artefacts.".format(batch.id, len(artefacts)))

print("PT: Starting data import processor for seed batch...")
batch.initJob()
batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
batch.job.save()

while batch.job.fetchLatestStatus() != collectionModels.BatchProcessingJob.Status.COMPLETED:
    time.sleep(0.5)
print("PT: Data import processor completed for seed batch.")

batch.reload()
for _, art in batch.artefacts.items():
    art.stage = collectionModels.BatchArtefact.Status.CONFIRMED
    art.save()

print("PT: All artefacts in seed batch marked as CONFIRMED (vetted).")

print("PT: Initializing integration stage for seed batch...")
batch.stage = collectionModels.Batch.Stage.INTEGRATION
batch.initJob()
batch.job.jobID = ThreadManager.defaultProcessor.addJob(HFCatalogueIntegrator.integrate, batch)
batch.save()

while batch.job.fetchLatestStatus() != collectionModels.BatchProcessingJob.Status.COMPLETED:
    time.sleep(0.5)
print("PT: Integration stage completed for seed batch.")

# Phase: Vetted Batch
print("PT: Uploading vetted batch artefacts...")
artefacts: list[artefactModels.Artefact] = []
for filename in FileOps.getFilenames("ptSeed2"):
    file = File(filename, "artefacts")
    
    # Save to local FM store first
    shutil.copy(os.path.join(os.getcwd(), "ptSeed2", filename), file.path())
    
    # Upload to cloud FM store
    res = FileManager.save(file=file)
    if isinstance(res, str):
        print("PT WARNING: Failed to upload vetted artefact file '{}' to cloud store; response: {}".format(filename, res))
        continue
    
    art = artefactModels.Artefact.fromFMFile(file)
    art.save()
    
    artefacts.append(art)

print("PT: Vetted artefacts uploaded and artefact records created.")

batch = collectionModels.Batch(user.id, "Vetted Batch", artefacts)
batch.stage = collectionModels.Batch.Stage.UNPROCESSED
batch.save()

print("PT: Vetted batch '{}' created with {} artefacts.".format(batch.id, len(artefacts)))

print("PT: Starting data import processor for vetted batch...")
batch.initJob()
batch.job.jobID = ThreadManager.defaultProcessor.addJob(DataImportProcessor.processBatch, batch)
batch.job.save()

while batch.job.fetchLatestStatus() != collectionModels.BatchProcessingJob.Status.COMPLETED:
    time.sleep(0.5)
print("PT: Data import processor completed for vetted batch.")

batch.reload()

for _, batchArt in batch.artefacts.items():
    batchArt.getArtefact(includeMetadata=True)
    art = batchArt.artefact
    if art.metadata and art.metadata.isHF():
        if art.name == "img1":
            art.metadata.raw.addInfo = LiteStore.read("img1")['text']
        elif art.name == "img2":
            art.metadata.raw.addInfo = LiteStore.read("img2")['text']
    
    art.save()
    batchArt.artefact = None
    batchArt.stage = collectionModels.BatchArtefact.Status.CONFIRMED
    batchArt.save()

print("PT: All artefacts in vetted batch updated with additional metadata and marked as CONFIRMED (vetted).")

print("PT: Completed presentation transformation process.")