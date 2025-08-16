from flask import Blueprint
from utils import JSONRes

apiBP = Blueprint('api', __name__, url_prefix='/')

@apiBP.route('/api/health', methods=['GET'])
def health():
    return JSONRes.new(200, "API is healthy.")