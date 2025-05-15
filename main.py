import os, sys, json
from flask import Flask, request, jsonify, url_for, render_template, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Welcome to ArchAIve!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)