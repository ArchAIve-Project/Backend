from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger
from decorators import enforceSchema, jsonOnly, checkAPIKey, Param
from schemas import Artefact
from sessionManagement import checkSession
from initemChat import InItemChatbot

chatbotBP = Blueprint('chatbot', __name__, url_prefix="/chatbot")

@chatbotBP.route('/query', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "history",
        lambda x: isinstance(x, list) and all(isinstance(item, dict) and "role" in item and "content" in item for item in x), None,
        invalidRes=JSONRes.new(400, "Invalid conversation history.", serialise=False)
    ),
    Param(
        "artefactID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Invalid artefact ID.", serialise=False)
    ),
    Param(
        "newPrompt",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Invalid new prompt.", ResType.USERERROR, serialise=False)
    )
)
@checkSession(strict=True)
def queryChatbot():
    history = request.json.get("history", [])
    artefactID = request.json.get("artefactID")
    newPrompt = request.json.get("newPrompt")

    try:
        targetArtefact = Artefact.load(artefactID, includeMetadata=True)
        if not isinstance(targetArtefact, Artefact):
            return JSONRes.new(404, "Artefact not found or invalid ID.")
    except Exception as e:
        Logger.log("CHATBOT QUERYCHATBOT ERROR: Unexpected error in loading artefact with ID '{}'; error: {}".format(artefactID, e))
        return JSONRes.ambiguousError()

    try:
        result = InItemChatbot.chat(targetArtefact, newPrompt, history)
        if isinstance(result, str) and result.startswith("ERROR:"):
            Logger.log("CHATBOT QUERYCHATBOT ERROR: Failed to generate response: {}".format(result))
            return JSONRes.ambiguousError()
    
        return JSONRes.new(200, "Chatbot response", data=result)
    except Exception as e:
        Logger.log("CHATBOT QUERYCHATBOT ERROR: {}".format(e))
        return JSONRes.ambiguousError()