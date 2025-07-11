import os, sys, json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, url_for, render_template, redirect
from flask_cors import CORS
from emailer import Emailer
from fm import FileManager, File
from ai import LLMInterface
from addons import ModelStore, ArchSmith
from services import Universal, Logger, ThreadManager, Encryption
from database import DI, DIError, JSONRes, ResType, FireConn
from fm import FileManager
from ccrPipeline import CCRPipeline
from NERPipeline import NERPipeline
from cnnclassifier import ImageClassifier
from captioner import ImageCaptioning, Vocabulary
from metagen import MetadataGenerator
from models import User, Artefact, Metadata

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
        autoLoad=False,
        ccr=CCRPipeline.loadChineseClassifier,
        ccrCharFilter=CCRPipeline.loadBinaryClassifier,
        ner=NERPipeline.load_model,
        cnn=ImageClassifier.load_model,
        imageCaptioner=ImageCaptioning.loadModel
    )
    
    # Setup LLMInterface
    res = LLMInterface.initDefaultClients()
    if res != True:
        print("MAIN BOOT: Failed to initialise default clients for LLMInterface; response:", res)
        sys.exit(1)
    print("LLMINTERFACE: Default clients initialised successfully.")
    
    # Import and register API blueprints
    from api import apiBP
    app.register_blueprint(apiBP)
    
    from identity import identityBP
    app.register_blueprint(identityBP)
    
    from cdn import cdnBP
    app.register_blueprint(cdnBP)
    
    # Debug execution
    if os.environ.get("DEBUG_MODE", "False") == "True":
        superuser = User.getSuperuser()
        if superuser == None:
            pwd = "123456"
            debugSuperuser = User("johndoe", "john@example.com", Encryption.encodeToSHA256(pwd), superuser=True)
            debugSuperuser.save()
            print("MAIN BOOT DEBUG: Superuser created with username '{}' and password '{}'.".format(debugSuperuser.username, pwd))

    print()
    print("MAIN BOOT: Pre-processing complete. Starting server...")
    print()

    app.run(host='0.0.0.0', port=8000)