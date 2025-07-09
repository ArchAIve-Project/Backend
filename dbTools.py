import time, pprint, datetime, os, sys, shutil, json
from services import Universal, ThreadManager, Encryption, Trigger
from firebase import FireConn
from models import DI, User, Artefact
from fm import File, FileManager, FileOps
from ai import LLMInterface, InteractionContext, Interaction
from addons import ModelStore, ArchSmith
from ccrPipeline import CCRPipeline
from NERPipeline import NERPipeline
from captioner import ImageCaptioning, Vocabulary
from cnnclassifier import ImageClassifier
from metagen import MetadataGenerator

class SetupStates:
    dbSetup = False
    fmSetup = False
    msSetup = False
    llmSetup = False

ArchSmith.setup()

def requireFC():
    if not FireConn.connected:
        print("FireConn Connection:", FireConn.connect())

def requireDI():
    if not SetupStates.dbSetup:
        print("DI Setup:", DI.setup())
        SetupStates.dbSetup = True

def requireFM():
    if not SetupStates.fmSetup:
        print("FileManager Setup:", FileManager.setup())
        SetupStates.fmSetup = True

def requireMS():
    if not SetupStates.msSetup:
        print("ModelStore Setup:", ModelStore.setup(
            ccr=CCRPipeline.loadChineseClassifier,
            ccrCharFilter=CCRPipeline.loadBinaryClassifier,
            imageCaptioner=ImageCaptioning.loadModel,
            cnn=ImageClassifier.load_model,
            ner=NERPipeline.load_model
        ))
        SetupStates.msSetup = True

def requireLLM():
    if not SetupStates.llmSetup:
        print("LLMInterface Default Clients Init:", LLMInterface.initDefaultClients())
        SetupStates.llmSetup = True

# Functions:
# 1. populate 2 users and 5 artefacts
# 2. create/delete user
# 3. create/delete artefact
# 4. process artefact and save
# 5. wipe DB
# 6. wipe FM
# 7. reset local data files

def populateDB():
    requireDI()
    requireFM()
    
    print()
    print("Populating database with 1 superuser, 4 regular users, and 5 dummy artefacts...")
    print()
    
    pwd = "123456"
    
    created: list[User] = []
    su = User.getSuperuser()
    if not su:
        su = User("johndoe", "john@example.com", Encryption.encodeToSHA256(pwd), superuser=True)
        su.save()
        created.append(su)
    
    users: list[User] = User.load()
    def userExists(username: str) -> bool:
        return any(user.username == username for user in users)
    
    if not userExists("sammyhopkins"):
        user2 = User("sammyhopkins", "sammy@example.com", Encryption.encodeToSHA256(pwd))
        user2.save()
        created.append(user2)

    if not userExists("jamieoliver"):    
        user3 = User("jamieoliver", "jamie@example.com", Encryption.encodeToSHA256(pwd))
        user3.save()
        created.append(user3)

    if not userExists("maryjane"):
        user4 = User("maryjane", "mary@example.com", Encryption.encodeToSHA256(pwd))
        user4.save()
        created.append(user4)

    if not userExists("peterparker"):
        user5 = User("peterparker", "peter@example.com", Encryption.encodeToSHA256(pwd))
        user5.save()
        created.append(user5)
    
    print("Created {} users:".format(len(created)))
    print("\tUsernames:", ", ".join(user.username if not user.superuser else "{} (SU)".format(user.username) for user in created))
    print("\tPassword for all users: '{}'".format(pwd))
    print()
    
    dummyArtefacts = ["hf1.png", "hf2.png", "mm1.jpg", "mm2.jpg", "mm3.jpg"]
    
    if not os.path.isdir("dummy"):
        raise Exception("Dummy directory not found. Please create a 'dummy' directory with the sample files.")
    
    # If not already present, copy to 'artefacts' dir (tracked by FM) and save
    currentArtefacts: list[Artefact] = Artefact.load()
    currentArtefactImages = [art.image for art in currentArtefacts]
    
    for filename in dummyArtefacts:
        file = File(filename, "artefacts")
        if not file.exists()[0]:
            src = os.path.join(os.getcwd(), "dummy", filename)
            shutil.copy(src, file.path())
            
            print("Copied file '{}' to artefacts. Save result: {}".format(filename, FileManager.save(file.store, file.filename)))
        
        if filename not in currentArtefactImages:
            art = Artefact(filename, filename, metadata=None)
            art.save()
            print("Saved artefact to DB: {}".format(art.name))
    
    print()
    print("5 artefacts populated:", ", ".join(dummyArtefacts))
    print()
    
    return True

def createUser():
    requireDI()
    
    print()
    print("Creating a new user...")
    print()
    
    user = User(
        username=input("Enter username: ").strip(),
        email=input("Enter email: ").strip(),
        pwd=Encryption.encodeToSHA256(input("Enter password: ").strip()),
        superuser=input("Is this a superuser? (yes/no): ").strip().lower() == "yes"
    )
    if input("Start a login session? (yes/no): ").strip().lower() == "yes":
        user.authToken = Universal.generateUniqueID(customLength=12)
        user.lastLogin = Universal.utcNowString()
    
    user.save()
    
    print()
    print("Created User:")
    print(user)
    print()
    return True

def deleteUser():
    requireDI()
    
    print()
    print("Deleting a user...")
    print()
    
    complete = False
    
    while not complete:
        identifier = input("Enter id, username, email or auth token of user to delete: ").strip()
        user: User | None = User.load(id=identifier) or User.load(username=identifier, email=identifier, authToken=identifier)
        if not user:
            complete = input("User not found. Try again? (yes/no): ").strip().lower() != "yes"
            continue
        
        user.destroy()
        print("Deleted user:", user.username)
        print()
        
        complete = True
    
    return True

def createArtefact():
    requireDI()
    requireFM()
    
    print()
    print("Creating a new artefact...")
    print()
    
    filename = input("Enter filename (with extension, should exist in ./artefacts): ").strip()
    name = input("Enter name (leave blank to set as filename w/o extension): ").strip() or filename.split(".")[0]
    print()
    file = File(filename, "artefacts")
    
    res = FileManager.save(file.store, file.filename)
    if isinstance(res, str):
        print("Error saving file:", res)
        print()
        return False
    print("File saved successfully:", res)
    
    art = Artefact(
        name=name,
        image=filename,
        metadata=None
    )
    art.save()
    
    print("Artefact object created:")
    print(art)
    print()
    
    return True

def deleteArtefact():
    requireDI()
    requireFM()
    
    print()
    print("Deleting an artefact...")
    print()
    
    complete = False
    
    while not complete:
        identifier = input("Enter id, name or image of artefact to delete: ").strip()
        artefact: Artefact | None = Artefact.load(id=identifier, name=identifier, image=identifier)
        if not artefact:
            complete = input("Artefact not found. Try again? (yes/no): ").strip().lower() != "yes"
            continue
        
        artefact.destroy()
        
        artFile = File(artefact.image, "artefacts")
        print("FileManager deletion result:", FileManager.delete(artFile))
        
        print("Deleted artefact:", artefact.name)
        print()
        
        complete = True
    
    return True

def processArtefact():
    requireDI()
    requireFM()
    requireMS()
    requireLLM()
    
    print()
    print("Triggering artefact processing workflow...")
    print()
    
    pass

def main(choices: list[int] | None=None):
    print("Welcome to ArchAIve DB Tools!")
    print("This is a debug script to quickly carry out common tasks.")
    print()
    
    if isinstance(choices, list):
        choice = choices.pop(0)
    else:
        choice = None
    
    while choice != 0:
        if choice is None:
            print("Choose an option:")
            print("1. Populate DB with 2 users and 5 artefacts")
            print("2. Create a new user")
            print("3. Delete a user")
            print("4. Create a new artefact")
            print("5. Delete an artefact")
            print("6. Process an artefact and save metadata")
            print("7. Wipe DB")
            print("8. Wipe FM")
            print("9. Reset local data files")
            print("0. Exit")
            print()
        
        try:
            choice = int(input("Enter choice: ")) if choice is None else choice
            if choice not in range(10):
                raise Exception()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except:
            print("Invalid choice. Please try again.")
            choice = None
            continue
        
        if choice == 0:
            print("Exiting...")
            sys.exit(0)
        elif choice == 1:
            populateDB()
        elif choice == 2:
            createUser()
        elif choice == 3:
            deleteUser()
        elif choice == 4:
            createArtefact()
        elif choice == 5:
            deleteArtefact()
        
        if isinstance(choices, list) and len(choices) > 0:
            choice = choices.pop(0)
        else:
            choice = None

if __name__ == "__main__":
    choices = None
    if len(sys.argv) > 1:
        choices = []
        for arg in sys.argv[1:]:
            choices.append(int(arg))
    print(choices)
    
    main(choices)