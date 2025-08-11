from flask import Blueprint, request
from utils import JSONRes
from services import Logger
from decorators import jsonOnly, enforceSchema, checkAPIKey
from sessionManagement import checkSession
from schemas import Artefact, User
from NERPipeline import NERPipeline
from addons import ArchSmith, ASReport

artBP = Blueprint('artefact', __name__, url_prefix="/artefact")

@artBP.route('/update', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("artefactID", str),
    ("name", str, None),
    ("tradCN", str, None),
    ("simplifiedCN", str, None),
    ("english", str, None),
    ("summary", str, None),
    ("caption", str, None),
    ("addInfo", str, None),
)
@checkSession(strict=True, provideUser=True)
def update(user: User):
    artefactID: str = request.json.get("artefactID")
    
    art = None
    try:
        art = Artefact.load(artefactID)
    except Exception as e:
        Logger.log("ARTEFACT UPDATE ERROR: Failed to load artefact {}: {}".format(artefactID, e))
        return JSONRes.ambiguousError()
    
    if not art.metadata:
        return JSONRes.new(400, "Target artefact does not have any metadata information to update.")
    
    changes = []
    
    # Artefact-related changes
    name: str = request.json.get("name")
        
    if art.name != name:
        art.name = name
        changes.append("Name")
    
    nerFailed = False
    
    # Metadata-related changes
    if art.metadata.isMM():
        tradCN: str | None = request.json.get("tradCN")
        simplifiedCN: str | None = request.json.get("simplifiedCN")
        english: str | None = request.json.get("english")
        summary: str | None = request.json.get("summary")
        
        if art.metadata.raw.tradCN != tradCN:
            art.metadata.raw.tradCN = tradCN
            changes.append('Traditional Chinese')
        if art.metadata.raw.simplifiedCN != simplifiedCN:
            art.metadata.raw.simplifiedCN = simplifiedCN
            changes.append('Simplified Chinese')
        if art.metadata.raw.english != english:
            art.metadata.raw.english = english
            changes.append('English')
            
            # NER Logic
            tracer = ArchSmith.newTracer(f"NER Labels regeneration for '{artefactID}'")
            try:
                newLabels = NERPipeline.predict(english, tracer)
                if isinstance(newLabels, str):
                    nerFailed = True
                else:
                    art.metadata.raw.nerLabels = newLabels
                    changes.append('NER Labels')
            except Exception as e:
                tracer.addReport(ASReport("ARTEFACT UPDATE ERROR", "Exception during NER prediction: {}".format(e)))
                nerFailed = True
            
        if art.metadata.raw.summary != summary:
            art.metadata.raw.summary = summary
            changes.append('Summary')
                        
            # Auto-generated description based on summary
            artDescription = (summary[:30] + "...") if summary and len(summary) > 30 else summary
            art.description = artDescription
            changes.append('Description')
                
    else:
        caption: str | None = request.json.get("caption")
        addInfo: str | None = request.json.get("addInfo")
        
        if art.metadata.raw.caption != caption:
            art.metadata.raw.caption = caption
            changes.append('Caption')
            
            # Auto-generated description based on caption
            artDescription = (caption[:30] + "...") if caption and len(caption) > 30 else caption
            art.description = artDescription
            changes.append('Description')
            
        if art.metadata.raw.addInfo != addInfo:
            art.metadata.raw.addInfo = addInfo
            changes.append('AddInfo')
    
    if len(changes) > 0:
        art.save()
        user.newLog("Artefact {} Update".format(artefactID), ", ".join(changes) + " updated.")
    else:
        return JSONRes.new(200, "No changes made to the artefact.")
    
    if nerFailed:
        return JSONRes.new(207, "Metadata updated successfully, but NER labelling failed.")
    
    return JSONRes.new(200, "Information updated successfully.")

@artBP.route('/removeFigure', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("artefactID", str),
    ("figureID", str)
)
@checkSession(strict=True, provideUser=True)
def removeFigure(user: User):
    artefactID: str = request.json.get("artefactID")
    figureID: str = request.json.get("figureID")
    
    art = None
    try:
        art = Artefact.load(artefactID)
    except Exception as e:
        Logger.log("ARTEFACT UPDATE ERROR: Failed to load artefact {}: {}".format(artefactID, e))
        return JSONRes.ambiguousError()
    
    if not art.metadata.isHF():
        return JSONRes.new(400, "Target artefact is not a human figure")
    
    if not art.metadata.raw.figureIDs:
        return JSONRes.new(400, "Target artefact does not have any human figure information to remove.")
    
    if figureID not in art.metadata.raw.figureIDs:
        return JSONRes.new(404, "Figure not found in artefact metadata.")
    
    art.metadata.raw.figureIDs.remove(figureID)
    
    try:
        art.save()
        user.newLog("Artefact {} Update".format(artefactID), "Figure {} removed.".format(figureID))
    except Exception as e:
        Logger.log("ARTEFACT REMOVEFIGURE ERROR: Failed to save artefact {} after removing figure ID: {}".format(artefactID, figureID))
        return JSONRes.ambiguousError()

    return JSONRes.new(200, f"Figure removed successfully.")