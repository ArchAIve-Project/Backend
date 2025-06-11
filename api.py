from flask import Blueprint, request, jsonify, render_template, redirect
from utils import JSONRes, ResType

apiBP = Blueprint('api', __name__, url_prefix='/api')

@apiBP.route('/health', methods=['GET'])
def health():
    return JSONRes.new(200, ResType.success, "API is healthy.")