import os, sys
from bootCheck import BootCheck
if not (len(sys.argv) > 1 and sys.argv[1] == "--skip-bootcheck"):
    BootCheck.check()
    print("BOOTCHECK: Check complete.")
del BootCheck

from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, url_for, render_template, redirect
from flask_cors import CORS
from emailer import Emailer
from fm import FileManager, File
from ai import LLMInterface
from addons import ModelStore, ArchSmith
from services import Universal, Logger, ThreadManager, Encryption, LiteStore
from firebase import FireConn
from database import DI, DIError
from fm import FileManager
from ccrPipeline import CCRPipeline
from NERPipeline import NERPipeline
from cnnclassifier import ImageClassifier
from captioner import ImageCaptioning, Vocabulary
from schemas import User

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, origins="*", supports_credentials=True, allow_private_network=True)

app.secret_key = os.environ['SECRET_KEY']

@app.route('/')
def home():
    return "Welcome to ArchAIve!"

if __name__ == "__main__":
    # Boot pre-processing
    ver = Universal.getVersion()
    if ver == None:
        print("MAIN BOOT: Unable to determine system version. Aborting.")
        sys.exit(1)
    
    print(f"MAIN BOOT: Starting ArchAIve v{ver}...")
    
    res = LiteStore.setup()
    if res != True:
        print("MAIN BOOT: Failed to setup LiteStore; response: {}".format(res))
        sys.exit(1)
    else:
        print("LITESTORE: Setup complete.")
    
    # Initialise database system
    if FireConn.checkPermissions():
        try:
            res = FireConn.connect()
            if res != True:
                raise Exception(res)
        except Exception as e:
            print(f"MAIN BOOT: Error during FireConn setup: {e}")
            sys.exit(1)
    
    try:
        res = DI.setup()
        if isinstance(res, DIError):
            raise Exception(res)
    except Exception as e:
        print(f"MAIN BOOT: Error during DI setup: {e}")
        sys.exit(1)
    
    # Setup FileManager
    if FireConn.connected:
        try:
            res = FileManager.setup()
            if res != True:
                raise Exception(res)
        except Exception as e:
            print(f"MAIN BOOT: Error during FileManager setup: {e}")
            sys.exit(1)
    else:
        print("MAIN BOOT WARNING: FireConn is not enabled; FileManager will not be available.")
    
    # Logging services
    Logger.setup()
    ArchSmith.setup()
    
    # Check Emailer context
    Emailer.checkContext()
    print("EMAILER: Context checked. Services enabled:", Emailer.servicesEnabled)
    
    # Background thread
    ThreadManager.initDefault()
    print("THREADMANAGER: Default background scheduler initialised.")
    
    # Setup ModelStore
    ModelStore.setup(
        autoLoad=os.environ.get("DEBUG_MS_AUTOLOAD", "True") == "True",
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier,
        cnn=ImageClassifier.load_model,
        imageCaptioner=ImageCaptioning.loadModel
    )
    
    # Load NERP model
    NERPipeline.load_model()
    
    # Setup LLMInterface
    res = LLMInterface.initDefaultClients()
    if res != True:
        print("MAIN BOOT: Failed to initialise default clients for LLMInterface; response:", res)
        sys.exit(1)
    print("LLMINTERFACE: Default clients initialised successfully.")
    
    # Import and register API blueprints
    
    ## Top-level routes
    from routes.api import apiBP
    app.register_blueprint(apiBP)
    
    from routes.cdn import cdnBP
    app.register_blueprint(cdnBP)
    
    # User management routes
    from routes.userManagement.auth import authBP
    app.register_blueprint(authBP)
    
    from routes.userManagement.userProfile import profileBP
    app.register_blueprint(profileBP)
    
    from routes.userManagement.admin import adminBP
    app.register_blueprint(adminBP)
    
    # Data Processing Routes
    from routes.dataProcessing.dataImport import dataImportBP
    app.register_blueprint(dataImportBP)
    
    # Data Studio Routes
    from routes.dataStudio.artefactManagement import artBP
    app.register_blueprint(artBP)
    
    # Superuser creation
    superuser = User.getSuperuser()
    if superuser == None:
        pwd = os.environ.get("SECRET_KEY", "123456")
        debugSuperuser = User("adminuser", "admin@email.com", Encryption.encodeToSHA256(pwd), "ArchAIve", "Admin", "Platform Superuser", superuser=True)
        debugSuperuser.save()
        print("MAIN BOOT: Superuser created with username '{}' and password '{}'.".format(debugSuperuser.username, pwd))

    print()
    print("MAIN BOOT: Pre-processing complete. Starting server...")
    print()

    app.run(host='0.0.0.0', port=8000)