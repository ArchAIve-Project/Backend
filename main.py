import os, sys, json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, url_for, render_template, redirect
from flask_cors import CORS
from services import Universal, Logger, Encryption
from database import DI, Ref, DIError, JSONRes, ResType, FireConn, FireRTDB

app = Flask(__name__)
CORS(app)

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
    
    # Logger service
    Logger.setup()
    
    # Background thread
    Universal.initAsync()
    
    # Import and register API blueprints
    from api import apiBP
    app.register_blueprint(apiBP)
    
    print()
    print("MAIN BOOT: Pre-processing complete. Starting server...")
    print()

    app.run(host='0.0.0.0', port=8000)