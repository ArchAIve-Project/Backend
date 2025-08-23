import os, sys
from bootCheck import BootCheck
if not (len(sys.argv) > 1 and sys.argv[1] == "--skip-bootcheck"):
    BootCheck.check()
    print("BOOTCHECK: Check complete.")
del BootCheck

from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request
from flask_cors import CORS
import appUtils
from emailer import Emailer
from fm import FileManager
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
appUtils.limiter.init_app(app)

app.secret_key = os.environ['SECRET_KEY']
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get("MAX_CONTENT_SIZE", 100)) * 1024 * 1024

@app.before_request
def beforeRequest():
    if LiteStore.read("systemLock") == True and (not request.path.startswith("/api")):
        return "ERROR: Service unavailable.", 503

@app.route('/')
def home():
    return "<h1>Welcome to the ArchAIve Backend System.</h1><p>Current version: {}</p><p>Current time: {}</p>".format(Universal.getVersion(), Universal.utcNowString(localisedTo=Universal.localisationOffset))

@app.errorhandler(413)
def requestEntityTooLarge(e):
    return "ERROR: Request entity too large.", 413

@app.errorhandler(429)
def tooManyRequests(e):
    return "ERROR: Too many requests.", 429

@app.errorhandler(500)
def internalServerError(e):
    return "ERROR: Internal server error.", 500

@app.errorhandler(503)
def serviceUnavailable(e):
    return "ERROR: Service unavailable.", 503

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
        autoLoad=os.environ.get("LLM_INFERENCE", "False") != "True",
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier,
        cnn=ImageClassifier.load_model,
        imageCaptionerResNet=ImageCaptioning.loadModel(weights='resnet'),
        imageCaptionerViT=ImageCaptioning.loadModel(weights='vit')
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

    # In Item Chatbot
    from routes.initemChatbot.chatbot import chatbotBP
    app.register_blueprint(chatbotBP)
    
    from routes.dataStudio.figureGallery import figureGalleryBP
    app.register_blueprint(figureGalleryBP)
    
    from routes.dataStudio.groupManagement import grpBP
    app.register_blueprint(grpBP)

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