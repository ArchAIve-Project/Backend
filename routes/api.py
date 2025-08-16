import os, json
from flask import Blueprint, request, render_template
from utils import JSONRes, ResType
from services import LiteStore, Logger, Encryption
from firebase import FireStorage
from fm import FileManager, File
from schemas import DI, User, Artefact, Category, Book, Batch

apiBP = Blueprint('api', __name__, url_prefix='/')

@apiBP.route('/api/health', methods=['GET'])
def health():
    return JSONRes.new(200, "API is healthy.")

@apiBP.route("/api/<key>/dir", methods=['GET'])
def dir(key):
    if not isinstance(key, str) or key != os.environ.get("SECRET_KEY"):
        return JSONRes.new(403, "Invalid or missing key.", ResType.USERERROR)
    
    return render_template("adminDir.html", lockStatus=LiteStore.read("systemLock") or False)

@apiBP.route("/api/<key>/toggleSystemLock", methods=['GET'])
def toggleSystemLock(key):
    if not isinstance(key, str) or key != os.environ.get("SECRET_KEY"):
        return JSONRes.new(403, "Invalid or missing key.", ResType.USERERROR)
    
    try:
        res = LiteStore.set("systemLock", (not LiteStore.read("systemLock")))
        if isinstance(res, str):
            raise Exception(res)
        
        newState = LiteStore.read("systemLock")
        Logger.log("API TOGGLESYSTEMLOCK: Toggled system lock to {}".format(newState))
        return JSONRes.new(200, "System lock toggled to {} successfully.".format(newState))
    except Exception as e:
        Logger.log("API TOGGLESYSTEMLOCK ERROR: Failed to toggle system lock; error: {}".format(e))
        return JSONRes.ambiguousError()

@apiBP.route("/api/<key>/retrieveLogs", methods=['GET'])
def retrieveLogs(key):
    if not isinstance(key, str) or key != os.environ.get("SECRET_KEY"):
        return JSONRes.new(403, "Invalid or missing key.", ResType.USERERROR)
    
    try:
        logs = Logger.readAll()
        if isinstance(logs, str):
            raise Exception(logs)
        
        return "<br>".join(logs)
    except Exception as e:
        Logger.log("API RETRIEVELOGS ERROR: Failed to retrieve logs; error: {}".format(e))
        return JSONRes.ambiguousError()

@apiBP.route("/api/<key>/retrieveAS", methods=['GET'])
def retrieveAS(key):
    if not isinstance(key, str) or key != os.environ.get("SECRET_KEY"):
        return JSONRes.new(403, "Invalid or missing key.", ResType.USERERROR)
    
    try:
        with open(os.path.join(os.getcwd(), "archsmith.json"), "r") as f:
            asData = json.load(f)
        if not asData:
            return JSONRes.new(404, "No ArchSmith data found.")
        if not isinstance(asData, dict):
            return JSONRes.new(500, "Invalid ArchSmith data format.")
        
        return asData
    except Exception as e:
        Logger.log("API RETRIEVEAS ERROR: Failed to retrieve ArchSmith data; error: {}".format(e))
        return JSONRes.ambiguousError()

@apiBP.route("/api/<key>/cleanSlateProtocol", methods=['GET'])
def cleanSlateProtocol(key):
    if not isinstance(key, str) or key != os.environ.get("SECRET_KEY"):
        return JSONRes.new(403, "Invalid or missing key.", ResType.USERERROR)

    try:
        DI.save(None)
        
        User("adminuser", "admin@email.com", Encryption.encodeToSHA256(os.environ.get("SECRET_KEY", "123456")), "ArchAIve", "Admin", "Platform Superuser", superuser=True).save()
        
        filenames = FireStorage.listFiles(namesOnly=True, ignoreFolders=True)
        files = [File(f.split('/')[1], f.split('/')[0]) for f in filenames]
        
        for file in files:
            FileManager.delete(file=file)
        
        with LiteStore.inMemoryStateLock:
            LiteStore.data = {}
            LiteStore.write()
        
        Logger.log("API CLEANSLATEPROTOCOL: Clean slate protocol executed successfully.")
        return JSONRes.new(200, "Clean slate protocol executed successfully.")
    except Exception as e:
        Logger.log("API CLEANSLATEPROTOCOL ERROR: Failed to execute clean slate protocol; error: {}".format(e))
        return JSONRes.ambiguousError()