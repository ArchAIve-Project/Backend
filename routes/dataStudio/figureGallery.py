from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger, LiteStore
from fm import FileManager, File
from decorators import jsonOnly, enforceSchema, checkAPIKey, Param
from schemas import Figure, User, Artefact
from sessionManagement import checkSession

figureGalleryBP = Blueprint('figureGallery', __name__, url_prefix="/gallery")

@figureGalleryBP.route('/updateFigure', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "figureID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Invalid figure ID.", serialise=False)
    ),
    Param(
        "label",
        lambda x: isinstance(x, str) and 1 < len(x) <= 20 and x.replace(' ', '').isalnum(),
        invalidRes=JSONRes.new(400, "Label must be alphanumeric and between 2 and 20 characters.", ResType.USERERROR, serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
def updateFigure(user: User):
    figureID: str = request.json.get("figureID")
    label: str = request.json.get("label").strip()
    
    targetFigure = None
    try:
        targetFigure: Figure = Figure.load(figureID)
        if not isinstance(targetFigure, Figure):
            return JSONRes.new(404, "Figure not found.")
    except Exception as e:
        Logger.log("FIGUREGALLERY UPDATEFIGURE ERROR: Couldn't load figure with ID '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()
    
    conflictingFigure = None
    try:
        conflictingFigure = Figure.load(label=label)
        if isinstance(conflictingFigure, Figure) and conflictingFigure.id != targetFigure.id:
            return JSONRes.new(409, "Figure with label '{}' already exists.".format(label), ResType.USERERROR)
    except Exception as e:
        Logger.log("FIGUREGALLERY UPDATEFIGURE ERROR: Couldn't load figure for conflict check with label '{}'; error: {}".format(label, e))
        return JSONRes.ambiguousError()
    
    prev = targetFigure.label
    targetFigure.label = label
    try:
        targetFigure.save(embeds=False)
        user.newLog("FIGURE UPDATE", "Updated label for figure '{}' from '{}' to '{}'.".format(figureID, prev, label))
        
        LiteStore.set("artefactMetadata", True)
        return JSONRes.new(200, "Figure updated successfully.")
    except Exception as e:
        Logger.log("FIGUREGALLERY UPDATEFIGURE ERROR: Couldn't save figure with ID '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()

@figureGalleryBP.route('/deleteFigure', methods=['POST'])
@checkAPIKey
@jsonOnly
@enforceSchema(
    Param(
        "figureID",
        lambda x: isinstance(x, str) and len(x) > 0,
        invalidRes=JSONRes.new(400, "Invalid figure ID.", serialise=False)
    )
)
@checkSession(strict=True, provideUser=True)
def deleteFigure(user: User):
    figureID: str = request.json.get("figureID")

    targetFigure = None
    try:
        targetFigure: Figure = Figure.load(figureID)
        
        if not isinstance(targetFigure, Figure):
            return JSONRes.new(404, "Figure not found.")
    except Exception as e:
        Logger.log("FIGUREGALLERY DELETEFIGURE ERROR: Couldn't load figure with ID '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()
    
    artefacts: list[Artefact] = []
    try:
        artefacts = Artefact.load()
    except Exception as e:
        Logger.log("FIGUREGALLERY DELETEFIGURE ERROR: Couldn't load artefacts; error: {}".format(e))
        return JSONRes.ambiguousError()
    
    for artefact in artefacts:
        if artefact.metadata and artefact.metadata.isHF() and isinstance(artefact.metadata.raw.figureIDs, list) and figureID in artefact.metadata.raw.figureIDs:
            try:
                artefact.metadata.raw.figureIDs.remove(figureID)
                artefact.metadata.save()
            except Exception as e:
                Logger.log("FIGUREGALLERY DELETEFIGURE ERROR: Couldn't remove figure ID '{}' from artefact '{}'; error: {}".format(figureID, artefact.id, e))
                return JSONRes.ambiguousError()
    
    try:
        file = File(targetFigure.headshot, "people")
        res = FileManager.delete(file=file)
        if isinstance(res, str):
            raise Exception(res)
    except Exception as e:
        Logger.log("FIGUREGALLERY DELETEFIGURE ERROR: Couldn't delete headshot for figure '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()
    
    try:
        targetFigure.destroy()
    except Exception as e:
        Logger.log("FIGUREGALLERY DELETEFIGURE ERROR: Couldn't delete figure with ID '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()
    
    try:
        user.newLog("FIGURE DELETE", "Deleted figure with label '{}' and ID '{}'.".format(targetFigure.label, figureID))
    except Exception as e:
        Logger.log("FIGUREGALLERY DELETEFIGURE WARNING: Failed to log deletion for figure '{}'; error: {}".format(figureID, e))
    
    LiteStore.set("artefactMetadata", True) 
    
    return JSONRes.new(200, "Figure deleted successfully.")