from flask import Blueprint, request
from initemChat import InItemChatbot
from sessionManagement import checkSession
from schemas import Artefact, User
from utils import JSONRes

chatbotBP = Blueprint('chatbot', __name__, url_prefix="/chatbot")

@chatbotBP.route('/query', methods=['POST'])
@checkSession(strict=True)
def queryChatbot():
    """
    Endpoint for querying the artefact chatbot.

    Expected JSON payload:
    {
        "artefactId": "abc123",
        "messagePrompt": "What's this artefact?",
        "conversationHistory": [{"role": "user", "content": "..."}, ...]
    }

    Returns:
    {
        "artefact": "abc123",
        "response": "Bot's reply",
        "history": [...]
    }
    """
    data = request.get_json()

    artefactID = data.get("artefactId")
    messagePrompt = data.get("messagePrompt")
    conversationHistory = data.get("conversationHistory")

    # Validate input
    if not artefactID or not messagePrompt or conversationHistory is None:
        return JSONRes.new(400, "ERROR: Missing required fields.")

    # Load artefact from DB
    artefact = Artefact.load(id=artefactID, includeMetadata=True)
    if not artefact:
        return JSONRes.new(404, "ERROR: Artefact {} not found.".format(artefactID))

    try:
        # Call the chatbot
        result = InItemChatbot.chat(artefact, messagePrompt, conversationHistory)
        if isinstance(result, str) and result.startswith("ERROR:"):
            return JSONRes.new(500, result)

        return JSONRes.new(200, "Chatbot response", data=result)
    except Exception as e:
        print("CHATBOT ERROR: {}".format(e))
        return JSONRes.new(500, "ERROR: Failed to run artefact chatbot.")