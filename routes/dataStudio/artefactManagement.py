from flask import Blueprint, request
from utils import JSONRes
from services import Logger
from decorators import jsonOnly, enforceSchema, checkAPIKey
from sessionManagement import checkSession
from schemas import Artefact, User
from NERPipeline import NERPipeline
from addons import ArchSmith, ASReport

artBP = Blueprint('artefact', __name__, url_prefix="/studio/artefact")

@artBP.route('/update', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    ("artefactID", str),
    ("name", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 25, None),
    ("tradCN", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 1000, None),
    ("simplifiedCN", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 1000, None),
    ("english", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 2000, None),
    ("summary", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 2000, None),
    ("caption", lambda x: isinstance(x, str) and 1 <= len(x.strip()) <= 1000, None),
    ("addInfo", lambda x: isinstance(x, str) and len(x.strip()) <= 3000, None),
)
@checkSession(strict=True, provideUser=True)
def update(user: User):
    """
    Update artefact metadata and related fields.

    This endpoint updates the artefact's basic info and metadata.  
    For MM-type metadata, if the English text changes, it triggers NER labelling,  
    and attempts to update NER labels. If NER labelling fails, the update still succeeds  
    but returns a partial success status.

    If summary or caption is updated without a provided description,  
    the artefact description is auto-generated as a truncated preview.

    Args:
        user (User): The authenticated user making the request.

    Request JSON Parameters:
        artefactID (str): ID of the artefact to update (required).
        name (str): New name of the artefact (optional).
        tradCN (str): Traditional Chinese metadata (optional, MM only).
        simplifiedCN (str): Simplified Chinese metadata (optional, MM only).
        english (str): English metadata text (optional, MM only).
        summary (str): Summary metadata (optional, MM only).
        caption (str): Caption metadata (optional, HF only).
        addInfo (str): Additional info metadata (optional, HF only).

    Returns:
        JSONRes with status code indicating success, partial success (207), or failure.
    """
    
    artefactID: str = request.json.get("artefactID")
    
    art = None
    try:
        art = Artefact.load(artefactID)
        
        if not isinstance(art, Artefact):
            return JSONRes.new(404, "Artefact not found.")
    except Exception as e:
        Logger.log("ARTEFACTMANAGEMENT UPDATE ERROR: Failed to load artefact {}: {}".format(artefactID, e))
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
            tracer = ArchSmith.newTracer("NER Labels regeneration for '{}'".format(artefactID))
            try:
                newLabels = NERPipeline.predict(english, tracer)
                if isinstance(newLabels, str):
                    nerFailed = True
                else:
                    art.metadata.raw.nerLabels = newLabels
                    changes.append('NER Labels')
            except Exception as e:
                tracer.addReport(ASReport("ARTEFACTMANAGEMENT UPDATE ERROR", "Exception during NER prediction: {}".format(e)))
                nerFailed = True
                
            tracer.end()
            ArchSmith.persist()
            
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
            changes.append('Additional Information')
    
    if len(changes) > 0:
        art.save()
        user.newLog("Artefact {} Update".format(art.name), ", ".join(changes) + " updated.")
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
    """
    Remove a specific figure ID from the human figure artefact's metadata.

    This endpoint removes a figure ID from the `figureIDs` list of a human figure (HF) artefact's metadata.  
    It verifies the artefact type, ensures the figure ID exists, removes it, and saves the artefact.

    Args:
        user (User): The authenticated user making the request.

    Request JSON Parameters:
        artefactID (str): ID of the artefact to modify (required).
        figureID (str): The figure ID to remove from the artefact's metadata (required).

    Returns:
        JSONRes indicating success or the nature of any failure.
    """
    
    artefactID: str = request.json.get("artefactID")
    figureID: str = request.json.get("figureID")
    
    art = None
    try:
        art = Artefact.load(artefactID)
        
        if not isinstance(art, Artefact):
            return JSONRes.new(404, "Artefact not found.")
    except Exception as e:
        Logger.log("ARTEFACTMANAGEMENT UPDATE ERROR: Failed to load artefact {}: {}".format(artefactID, e))
        return JSONRes.ambiguousError()
    
    if art.artType != 'hf':
        return JSONRes.new(400, "Target artefact is not a human figure")
    
    if not art.metadata.raw.figureIDs:
        return JSONRes.new(400, "Target artefact does not have any human figure information to remove.")
    
    if figureID not in art.metadata.raw.figureIDs:
        return JSONRes.new(404, "Figure not found in artefact metadata.")
    
    art.metadata.raw.figureIDs.remove(figureID)
    
    try:
        art.metadata.save()
        user.newLog("Artefact {} Update".format(art.name), "Figure {} removed.".format(figureID))
    except Exception as e:
        Logger.log("ARTEFACTMANAGEMENT REMOVEFIGURE ERROR: Failed to save artefact {} after removing figure ID: {}".format(artefactID, figureID))
        return JSONRes.ambiguousError()

    return JSONRes.new(200, "Figure removed successfully.")