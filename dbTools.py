import time, pprint, datetime, os, sys, shutil, json
from services import Universal, ThreadManager, Encryption, Trigger
from firebase import FireConn
from models import DI, User, Artefact, Metadata, Category, Book, CategoryArtefact
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
    requireFC()
    if not SetupStates.dbSetup:
        print("DI Setup:", DI.setup())
        SetupStates.dbSetup = True

def requireFM():
    requireFC()
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
# 1. Populate DB with 2 users and 5 artefacts
# 2. Create a new user
# 3. Delete a user
# 4. Create a new artefact
# 5. Delete an artefact
# 6. Process an artefact and save metadata
# 7. Wipe DB
# 8. Wipe FM
# 9. Reset local data files
# 10. Remove categories
# 11. Populate more data (after 1)
# 0: Exit

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
        if not file.exists()[1]:
            src = os.path.join(os.getcwd(), "dummy", filename)
            shutil.copy(src, file.path())
            
            print("Copied file '{}' to artefacts. Save result: {}".format(filename, FileManager.save(file.store, file.filename)))
        
        description = "Description for artefact {}".format(filename)
        
        if filename not in currentArtefactImages:
            art = Artefact(filename, filename, description=description, metadata=None)
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
        print("FileManager deletion result:", FileManager.delete(file=artFile))
        
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
    
    identifier = input("Enter id, name or image of artefact to process: ").strip()
    artefact: Artefact | None = Artefact.load(id=identifier, name=identifier, image=identifier)
    while not artefact:
        tryAgain = input("Artefact not found. Try again? (yes/no): ").strip().lower() == "yes"
        if not tryAgain:
            print("Exiting artefact processing.")
            print()
            return False
        
        identifier = input("Enter id, name or image of artefact to process: ").strip()
        artefact: Artefact | None = Artefact.load(id=identifier, name=identifier, image=identifier)
    
    file = artefact.getFMFile()
    res = FileManager.prepFile(file=file)
    if isinstance(res, str):
        print("Failed to prepare file locally; response:", res)
        print("Exiting artefact processing.")
        print()
        return False
    
    print()
    print("Processing artefact... this may take a while")
    print()
    output = MetadataGenerator.generate(file.path())
    if isinstance(output, str):
        print("Error processing artefact:", output)
        print("Exiting artefact processing.")
        print()
        return False
    
    artefact.metadata = Metadata.fromMetagen(artefact.id, output)
    artefact.save()
    
    print("Artefact processed and saved successfully. Artefact data:")
    print(artefact)
    print()
    
    return True

def wipeDB():
    requireDI()
    
    print()
    print("Wiping database...")
    print()
    
    if input("Are you sure you want to wipe the database? This cannot be undone. (yes/no): ").strip().lower() != "yes":
        print("Database wipe cancelled.")
        return False
    
    DI.save(None)
    print()
    print("Database wiped successfully.")
    print()
    
    return True

def wipeFM():
    requireFM()
    
    print()
    print("Wiping FileManager...")
    print()
    
    if input("Are you sure you want to wipe the FileManager? This cannot be undone. (yes/no): ").strip().lower() != "yes":
        print("FileManager wipe cancelled.")
        return False
    
    print("Delete All Result:", FileManager.deleteAll())
    
    print()
    print("FileManager wiped successfully.")
    print()
    
    return True

def resetLocalDataFiles():
    print()
    print("Resetting local persistent stores...")
    print()
    
    if input("Are you sure you want to reset local data files? This will delete all local data. (yes/no): ").strip().lower() != "yes":
        print("Local data reset cancelled.")
        return False
    
    try:
        for store in FileManager.stores:
            shutil.rmtree(os.path.join(os.getcwd(), store), ignore_errors=True)
        
        os.remove("database.json")
        os.remove("fmContext.json")
        os.remove("archsmith.json")
    except Exception as e:
        print("Error resetting local data files:", e)
        print("Local data reset failed.")
        return False
    
    print()
    print("All local persistent stores deleted.")
    print("The DB tools context is now invalid, and thus DB tools will exit.")
    print()
    
    sys.exit(0)

def removeCategories():
    requireDI()

    categories = [
        "Meeting Minutes Grp1",
        "Meeting Minutes Grp2",
    ]

    for cat_name in categories:
        cat = Category.load(name=cat_name)
        if cat:
            cat.destroy()
            print(f"Category '{cat.name}' removed from the database.")
        else:
            print(f"Category '{cat_name}' not found.")
            
def duplicateDummyImages(folder="./dummy", out_folder="./moreDummy"):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # HF duplication
    hf_sources = [os.path.join(folder, "hf{}.png".format(i)) for i in range(1, 3)]
    for i in range(3, 51):
        src = hf_sources[(i - 3) % 2]
        dst = os.path.join(out_folder, "hf{}.png".format(i))
        shutil.copyfile(src, dst)
        print("Created {} from {}".format(dst, os.path.basename(src)))

    # MM duplication
    mm_sources = [os.path.join(folder, "mm{}.jpg".format(i)) for i in range(1, 4)]
    for i in range(4, 51):
        src = mm_sources[(i - 4) % 3]
        dst = os.path.join(out_folder, "mm{}.jpg".format(i))
        shutil.copyfile(src, dst)
        print("Created {} from {}".format(dst, os.path.basename(src)))

def populateMoreArtefacts():
    requireDI()
    requireFM()

    print("\nPreparing moreDummy folder...\n")
    moreDummyPath = os.path.join(os.getcwd(), "moreDummy")
    if not os.path.exists(moreDummyPath):
        os.makedirs(moreDummyPath)

    # Check if 50 hf and 50 mm images are present
    hf_files = ["hf{}.png".format(i) for i in range(1, 51)]
    mm_files = ["mm{}.jpg".format(i) for i in range(1, 51)]
    existing_files = set(os.listdir(moreDummyPath))

    hf_count = sum(1 for f in hf_files if f in existing_files)
    mm_count = sum(1 for f in mm_files if f in existing_files)

    if hf_count < 50 or mm_count < 50:
        print("Not enough HF/MM images found in 'moreDummy'. Generating from './dummy'...")
        duplicateDummyImages(folder="./dummy", out_folder=moreDummyPath)
    else:
        print("Found 50 HF and 50 MM images in moreDummy.")

    currentArtefacts = Artefact.load()
    currentArtefactImages = {art.image for art in currentArtefacts}

    # Create artefacts
    more_mm = ["mm{}.jpg".format(i) for i in range(4, 51)]
    more_hf = ["hf{}.png".format(i) for i in range(3, 51)]
    allFiles = more_mm + more_hf
    newlyAdded = []

    for filename in allFiles:
        file = File(filename, "artefacts")
        target_path = file.path()
        source_path = os.path.join(moreDummyPath, filename)

        if not os.path.isfile(target_path):
            shutil.copy(source_path, target_path)
            FileManager.save(file.store, file.filename)
            print("Copied '{}' to artefacts.".format(filename))

        if filename not in currentArtefactImages:
            description = "Description for artefact {}".format(filename)
            art = Artefact(filename, filename, description=description, metadata=None)
            art.save()
            newlyAdded.append(filename)
            print("Saved artefact to DB: {}".format(filename))

    print("\n{} artefacts populated: {}\n".format(len(newlyAdded), ", ".join(newlyAdded)))

    # Create categories for human figures
    print("Adding categories for human figures...")
    all_artefacts = Artefact.load()
    image_to_id = {art.image: art.id for art in all_artefacts}
    hf_all = ["hf{}.png".format(i) for i in range(1, 51)]

    categories = [
        ("Human Group 1", hf_all[:30]),
        ("Human Group 2", hf_all[30:45]),
        ("Human Group 3", hf_all[45:])
    ]

    for cat_name, artefact_filenames in categories:
        cat = Category.load(name=cat_name)
        if not cat:
            cat = Category(name=cat_name, description="Category for {}".format(cat_name.lower()))

        added = 0
        for filename in artefact_filenames:
            artefact_id = image_to_id.get(filename)
            if not artefact_id:
                print("Artefact with image '{}' not found in DB. Skipping.".format(filename))
                continue

            try:
                cat.add(artefact_id, reason="Auto-assigned to {}".format(cat_name))
                added += 1
            except Exception as e:
                print("Failed to add '{}' to '{}': {}".format(filename, cat_name, e))

        cat.save()
        print("Category '{}' saved with {}/{} artefacts.".format(cat.name, added, len(artefact_filenames)))

    # Create 10 books using mm1.jpg to mm50.jpg, but store artefact IDs, not filenames
    print("\nPopulating database with 10 books using mm1.jpg to mm50.jpg...\n")
    mm_all = ["mm{}.jpg".format(i) for i in range(1, 51)]
    mm_counts = [15, 10, 5, 5, 5, 5, 4, 3, 1, 0]
    created = []
    offset = 0

    for i, count in enumerate(mm_counts):
        title = "SCCCI Minutes Volume {}".format(i + 1)
        subtitle = "Auto-generated Volume {}".format(i + 1)
        mm_filenames = mm_all[offset: offset + count]
        offset += count

        # Convert mm filenames to artefact IDs
        mmIDs = []
        for mm_file in mm_filenames:
            artefact_id = image_to_id.get(mm_file)
            if artefact_id:
                mmIDs.append(artefact_id)
            else:
                print("Warning: Artefact ID for '{}' not found, skipping in book '{}'.".format(mm_file, title))

        existing_book = Book.load(title=title)
        if not existing_book:
            book = Book(title=title, subtitle=subtitle, mmIDs=mmIDs)
            book.save()
            created.append(book)
            print("Created Book: '{}' with {} MMIDs".format(book.title, len(mmIDs)))
        else:
            print("Book already exists: '{}'".format(existing_book.title))

    print("\n{} book(s) created.\n".format(len(created)))
    return True

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
            print("10. Remove categories")
            print("11. Populate more data (after 1)")
            print("0: Exit")
        
        try:
            choice = int(input("Enter choice: ")) if choice is None else choice
            if choice not in range(12):
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
        elif choice == 6:
            processArtefact()
        elif choice == 7:
            wipeDB()
        elif choice == 8:
            wipeFM()
        elif choice == 9:
            resetLocalDataFiles()
        elif choice == 10:
            removeCategories()
        elif choice == 11:
            populateMoreArtefacts()
        
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