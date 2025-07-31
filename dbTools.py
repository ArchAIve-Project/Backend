import time, pprint, datetime, os, sys, shutil, json, random
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
# 12. Populate 50 HF and 3 Cat (20 ungrouped HF, 1 empty Cat)
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

def removeCategories(cat_name):
    requireDI()
    print()
    print("Attempting to remove category: {}".format(cat_name))

    try:
        cat = Category.load(name=cat_name)
    except Exception as e:
        print("Error loading category '{}': {}".format(cat_name, str(e)))
        return False

    if cat:
        try:
            cat.destroy()
            print("Category '{}' removed from the database.".format(cat.name))
            return True
        except Exception as e:
            print("Error deleting category '{}': {}".format(cat.name, str(e)))
            return False
    else:
        print("Category '{}' not found.".format(cat_name))
        return False
            
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
            print(f"Copied '{filename}' to artefacts.")

        if filename not in currentArtefactImages:
            # Longer, realistic descriptions for meeting minutes (MM)
            if filename.startswith('mm'):
                description = None

                metadata = {
                    "traditional_chinese": (
                        "本文件為名為 {} 的會議記錄檔案，內容詳實地記載了整場會議的重點討論與決策結果。\n"
                        "本次會議涵蓋多個重要議題，包括跨部門專案協作進度更新、預算調整審議、時間表重新規劃、以及下一季度的業務策略方向。\n"
                        "會議期間亦針對人力資源調配、風險管理措施與品質保證流程進行了深入探討與共識建構。\n"
                        "所有與會者，包括執行長、副總經理、各專案主管與主要利害關係人，皆積極參與並提供關鍵意見，形成最終決策依據。\n"
                        "此記錄將作為後續追蹤與專案審查的依據，亦利於透明化溝通與跨部門協同運作。"
                    ).format(filename),

                    "simplified_chinese": (
                        "本文件为名为 {} 的会议纪要文件，详细记录了会议期间的关键讨论内容与决策成果。\n"
                        "会议主题涵盖跨部门项目协作的最新进展、预算调整的审批流程、整体时间线的优化建议，以及下一季度业务战略的方向规划。\n"
                        "此外，参会人员还对人力资源配置、风险控制策略及质量管理体系进行了充分讨论，并形成初步共识。\n"
                        "本次会议由公司高管层、项目主管、财务负责人及关键利益相关方共同参与，确保所有观点均被纳入考量。\n"
                        "会议纪要不仅有助于后续的工作执行与评估，还将作为战略决策档案予以保存与回溯。"
                    ).format(filename),

                    "english_translation": (
                        "This document contains the meeting minutes titled 'Board Quarterly Strategy Meeting', meticulously capturing the core discussions, strategic decisions, and action-oriented outcomes of the session held on March 15th, 2024. "
                        "Topics addressed include cross-departmental project updates, fiscal budget revisions, timeline rescheduling, and strategic planning for the next quarter starting April 1st, 2024. "
                        "Participants such as the CEO, Deputy General Manager, Project Directors, and Finance Department actively contributed to discussions around workforce allocation, risk management, and quality assurance protocols. "
                        "The meeting took place at the New York headquarters, bringing together key stakeholders from various locations including Chicago, San Francisco, and Boston. "
                        "Additionally, the Global Partnership Summit in Washington D.C., scheduled for June 2024, was referenced multiple times as an important upcoming event involving representatives from international organizations and government agencies. "
                        "The team also discussed plans to attend the Annual Tech Expo in Las Vegas and the Environmental Forum in Seattle, both recognized as major events shaping industry trends. "
                        "This record serves as a comprehensive reference for project follow-up, audits, strategic decision-making, and interdepartmental coordination."
                    ).format(filename),

                    "summary": (
                        "The meeting minutes titled 'Board Quarterly Strategy Meeting' detail cross-functional collaboration updates, strategic decision-making by executives, and future-oriented planning. "
                        "Key agenda items include budget approvals, timeline realignments, human resource adjustments, and stakeholder coordination."
                    ).format(filename),

                    "ner_labels": [
                        ["Board Quarterly Strategy Meeting", "ORGANIZATION"],
                        ["March 15th, 2024", "DATE"],
                        ["next quarter", "DATE"],
                        ["April 1st, 2024", "DATE"],
                        ["June 2024", "DATE"],
                        ["CEO", "PERSON"],
                        ["Deputy General Manager", "PERSON"],
                        ["Project Directors", "PERSON"],
                        ["Finance Department", "ORGANIZATION"],
                        ["New York", "GPE"],
                        ["Chicago", "GPE"],
                        ["San Francisco", "GPE"],
                        ["Boston", "GPE"],
                        ["Washington D.C.", "LOCATION"],
                        ["Global Partnership Summit", "EVENT"],
                        ["Annual Tech Expo", "EVENT"],
                        ["Environmental Forum", "EVENT"],
                        ["key stakeholders", "PERSON"],
                        ["international organizations", "ORGANIZATION"],
                        ["government agencies", "ORGANIZATION"]
                    ],

                    "correction_applied": True,
                    "pre_correction_accuracy": "94.73%",
                    "post_correction_accuracy": "99.14%"
                }
            
            else:
                # Longer, realistic descriptions for human figures (HF)
                description = None

                metadata = {
                    'figureIDs': ["fig-{}-{}".format(filename, i) for i in range(1, random.randint(3, 6))],
                    
                    'caption': (
                        "A high-resolution group photograph titled '{}', portraying a natural and dynamic interaction among multiple individuals. "
                        "The subjects are captured in various candid poses, contributing to the realism and diversity of the scene. "
                        "Environmental factors such as lighting, background elements, and occlusions add complexity to the image, making it suitable for training advanced vision systems."
                    ).format(filename),
                    
                    'addInfo': (
                        "This image belongs to an extensive dataset curated specifically for developing and benchmarking human figure detection, recognition, and tracking algorithms. "
                        "Each individual in the scene exhibits varied gestures, apparel styles, and positioning to simulate real-world variability. "
                        "The dataset emphasizes diversity across demographic groups and scene contexts—indoor, outdoor, formal, and casual settings. "
                        "Captured using high-quality sensors under fluctuating environmental conditions, this image aids in robust model generalization. "
                        "Annotations associated with the image include bounding boxes, body keypoints, and instance-level IDs, facilitating multi-task training pipelines. "
                        "The photo also serves secondary purposes such as pose estimation, social interaction modeling, and occlusion-resilient recognition tasks."
                    )
                }
                
            art = Artefact(
                name=filename,
                image=filename,
                metadata=metadata,
                description=description
            )
            art.save()
            newlyAdded.append(filename)
            print(f"Saved artefact to DB with metadata: {filename}")

    print(f"\n{len(newlyAdded)} artefacts populated: {', '.join(newlyAdded)}\n")

    # Create categories for human figures (unchanged)
    print("Adding categories for human figures...")
    all_artefacts = Artefact.load()
    image_to_id = {art.image: art.id for art in all_artefacts}
    hf_all = ["hf{}.png".format(i) for i in range(1, 51)]

    categories = [
        (
            "Human Group 1",
            hf_all[:30],
            "This collection comprises images of diverse human groups captured in various settings, focusing on interactions, body language, and group dynamics. "
            "The photos showcase people engaged in collaborative activities, casual conversations, and everyday social scenarios, providing a rich dataset for analyzing interpersonal behavior and group composition."
        ),
        (
            "Human Group 2",
            hf_all[30:45],
            "Featuring a varied set of images, this category emphasizes mid-sized groups of individuals in both indoor and outdoor environments. "
            "The subjects display a range of emotions and postures, contributing to the dataset’s diversity in social contexts and physical arrangements, essential for training robust human detection and recognition models."
        ),
        (
            "Human Group 3",
            hf_all[45:],
            "A specialized collection capturing larger human assemblies in dynamic environments such as events, gatherings, and public spaces. "
            "These images present challenges such as occlusion, varied lighting, and complex spatial relationships among individuals, making them valuable for advanced research in crowd analysis, pose estimation, and multi-person tracking."
        )
    ]

    for cat_name, artefact_filenames, cat_description in categories:
        cat = Category.load(name=cat_name)
        if not cat:
            cat = Category(name=cat_name, description=cat_description)

        added = 0
        for filename in artefact_filenames:
            artefact_id = image_to_id.get(filename)
            if not artefact_id:
                print(f"Artefact with image '{filename}' not found in DB. Skipping.")
                continue

            try:
                cat.add(artefact_id, reason="Auto-assigned to {}".format(cat_name))
                added += 1
            except Exception as e:
                print(f"Failed to add '{filename}' to '{cat_name}': {e}")

        cat.save()
        print(f"Category '{cat.name}' saved with {added}/{len(artefact_filenames)} artefacts.")
        
    # Create an empty category (unchanged)
    print("\nCreating an empty category...\n")
    empty_cat_name = "Empty Category"
    empty_category = Category.load(name=empty_cat_name)
    if not empty_category:
        empty_category = Category(name=empty_cat_name, description="This is an empty category with no artefacts.")
        empty_category.save()
        print(f"Empty category '{empty_cat_name}' created.")
    else:
        print(f"Empty category '{empty_cat_name}' already exists.")

    # Create 10 books (unchanged)
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

        mmIDs = []
        for mm_file in mm_filenames:
            artefact_id = image_to_id.get(mm_file)
            if artefact_id:
                mmIDs.append(artefact_id)
            else:
                print(f"Warning: Artefact ID for '{mm_file}' not found, skipping in book '{title}'.")

        existing_book = Book.load(title=title)
        if not existing_book:
            book = Book(title=title, subtitle=subtitle, mmIDs=mmIDs)
            book.save()
            created.append(book)
            print(f"Created Book: '{book.title}' with {len(mmIDs)} MMIDs")
        else:
            print(f"Book already exists: '{existing_book.title}'")

    print(f"\n{len(created)} book(s) created.\n")
    return True

def populateHFCat():
    requireDI()
    requireFM()

    def duplicateHFImages(folder, out_folder, prefix="hf", ext=".png", count=50):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        source_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if not source_files:
            print(f"No source images found in '{folder}' to duplicate.")
            return

        for i in range(count):
            src_file = random.choice(source_files)
            src_path = os.path.join(folder, src_file)
            dst_filename = f"{prefix}{i + 1}{ext}"
            dst_path = os.path.join(out_folder, dst_filename)
            shutil.copy(src_path, dst_path)
            print(f"Created {dst_path} from {src_file}")

    print("\nPreparing hfDummy folder...\n")
    hfDummyPath = os.path.join(os.getcwd(), "hfDummy")
    if not os.path.exists(hfDummyPath):
        os.makedirs(hfDummyPath)

    hf_files = [f"hf{i}.png" for i in range(1, 51)]
    existing_files = set(os.listdir(hfDummyPath))
    hf_count = sum(1 for f in hf_files if f in existing_files)

    if hf_count < 50:
        print("Not enough HF images found in 'hfDummy'. Generating from './dummy'...")
        duplicateHFImages(folder="./dummy", out_folder=hfDummyPath)
    else:
        print("Found 50 HF images in hfDummy.")

    currentArtefacts = Artefact.load()
    currentArtefactImages = {art.image for art in currentArtefacts}

    newlyAdded = []
    for filename in hf_files:
        file = File(filename, "artefacts")
        target_path = file.path()
        source_path = os.path.join(hfDummyPath, filename)

        if not os.path.isfile(source_path):
            print(f"Source file missing: {source_path}. Skipping.")
            continue

        if not os.path.isfile(target_path):
            shutil.copy(source_path, target_path)
            FileManager.save(file.store, file.filename)
            print(f"Copied '{filename}' to artefacts.")

        if filename not in currentArtefactImages:
            description = None
            
            metadata = {
                'figureIDs': ["fig-{}-{}".format(filename, i) for i in range(1, random.randint(3, 6))],
                
                'caption': (
                    "A high-resolution group photograph titled '{}', portraying a natural and dynamic interaction among multiple individuals. "
                    "The subjects are captured in various candid poses, contributing to the realism and diversity of the scene. "
                    "Environmental factors such as lighting, background elements, and occlusions add complexity to the image, making it suitable for training advanced vision systems."
                ).format(filename),
                
                'addInfo': (
                    "This image belongs to an extensive dataset curated specifically for developing and benchmarking human figure detection, recognition, and tracking algorithms. "
                    "Each individual in the scene exhibits varied gestures, apparel styles, and positioning to simulate real-world variability. "
                    "The dataset emphasizes diversity across demographic groups and scene contexts—indoor, outdoor, formal, and casual settings. "
                    "Captured using high-quality sensors under fluctuating environmental conditions, this image aids in robust model generalization. "
                    "Annotations associated with the image include bounding boxes, body keypoints, and instance-level IDs, facilitating multi-task training pipelines. "
                    "The photo also serves secondary purposes such as pose estimation, social interaction modeling, and occlusion-resilient recognition tasks."
                )
            }
            
            art = Artefact(
                name=filename,
                image=filename,
                metadata=metadata,
                description=description
            )
            art.save()
            newlyAdded.append(filename)
            print(f"Saved artefact to DB with metadata: {filename}")

    print(f"\n{len(newlyAdded)} HF artefacts populated.\n")

    # Assign categories with enhanced descriptions
    all_artefacts = Artefact.load()
    image_to_id = {art.image: art.id for art in all_artefacts}
    hf_all = [f"hf{i}.png" for i in range(1, 51)]

    categories = [
        (
            "Business Meetings",
            hf_all[:20],
            "A curated selection of high-quality images capturing formal business meetings, corporate conferences, and professional events. "
            "These photos highlight various aspects such as executive discussions, boardroom interactions, presentations, and strategic planning sessions. "
            "The collection emphasizes professional attire, focused expressions, and collaborative environments, providing valuable visual context for training models related to workplace behavior and formal group dynamics."
        ),
        (
            "Social Gatherings",
            hf_all[20:30],
            "This collection showcases candid and spontaneous moments from informal social gatherings, networking events, and casual meet-ups among friends and colleagues. "
            "Images portray a wide range of social interactions, including conversations, laughter, and group activities in diverse settings such as cafes, outdoor parks, and private parties. "
            "The variety in lighting, poses, and group compositions offers rich material for training systems focused on social behavior analysis and interpersonal dynamics."
        ),
        (
            "Special Events",
            [],
            "An exclusive category reserved for upcoming collections of special occasions including award ceremonies, celebrations, gala dinners, and cultural festivals. "
            "These images are expected to feature a blend of formal and festive atmospheres, capturing both the grandeur of event settings and the emotional moments of participants. "
            "Once populated, this category will serve as a resource for models requiring contextual understanding of ceremonial events and large-scale social functions."
        )
    ]

    for cat_name, artefact_filenames, cat_desc in categories:
        cat = Category.load(name=cat_name)
        if not cat:
            cat = Category(name=cat_name, description=cat_desc)

        added = 0
        for filename in artefact_filenames:
            artefact_id = image_to_id.get(filename)
            if not artefact_id:
                print(f"Artefact with image '{filename}' not found. Skipping.")
                continue

            try:
                cat.add(artefact_id, reason=f"Auto-assigned to {cat_name}")
                added += 1
            except Exception as e:
                print(f"Failed to add '{filename}' to '{cat_name}': {e}")

        cat.save()
        print(f"Category '{cat.name}' saved with {added}/{len(artefact_filenames)} artefacts.")

    print("\nHF-only population complete with enriched metadata.\n")
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
            print("12: Populate 50 HF and 3 Cat (20 ungrouped HF, 1 empty Cat)")
            print("0: Exit")
        
        try:
            choice = int(input("Enter choice: ")) if choice is None else choice
            if choice not in range(13):
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
            print()
            print("Enter category names to remove. Type 'done' to finish.")
            while True:
                name = input("Category name: ").strip()
                if name.lower() == "done":
                    break
            removeCategories(name)
        elif choice == 11:
            populateMoreArtefacts()
        elif choice == 12:
            populateHFCat()
        
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