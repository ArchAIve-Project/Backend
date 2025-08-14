from flask import Blueprint, request
from utils import JSONRes, ResType
from services import Logger, LiteStore
from decorators import jsonOnly, enforceSchema, checkAPIKey, Param
from schemas import Figure, User
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
            return JSONRes.new(409, "Figure with label '{}' already exists.".format(label))
    except Exception as e:
        Logger.log("FIGUREGALLERY UPDATEFIGURE ERROR: Couldn't load figure for conflict check with label '{}'; error: {}".format(label, e))
        return JSONRes.ambiguousError()
    
    prev = targetFigure.label
    targetFigure.label = label
    try:
        targetFigure.save(embeds=False)
        user.newLog("FIGURE UPDATE", "Updated label for figure '{}' from '{}' to '{}'.".format(figureID, prev, label))
        return JSONRes.new(200, "Figure updated successfully.")
    except Exception as e:
        Logger.log("FIGUREGALLERY UPDATEFIGURE ERROR: Couldn't save figure with ID '{}'; error: {}".format(figureID, e))
        return JSONRes.ambiguousError()