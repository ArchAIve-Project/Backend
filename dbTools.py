import time, pprint, datetime, os, sys, shutil, json
from services import Universal, ThreadManager, Encryption, Trigger

# Functions:
# 1. populate 2 users and 5 artefacts
# 2. create/delete user
# 3. create/delete artefact
# 4. process artefact and save
# 5. wipe DB
# 6. wipe FM
# 7. reset local data files

def populate_db():
    print("Populating database with 1 superuser, 4 regular users, and 5 dummy artefacts...")
    print()
    
    from models import DI, User, Artefact
    from fm import FileManager, File
    DI.setup()
    FileManager.setup()
    
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

def main(choice: int=None):
    print("Welcome to ArchAIve DB Tools!")
    print("This is a debug script to quickly carry out common tasks.")
    print()
    
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
            populate_db()
        
        choice = None

if __name__ == "__main__":
    choice = None
    if len(sys.argv) > 1:
        choice = int(sys.argv[1])
    
    main(choice)