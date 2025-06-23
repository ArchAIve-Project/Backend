import os, json, datetime
from flask import Flask, render_template, request, jsonify
from addons import ASTracer

def getTracers():
    with open(os.path.join(os.path.dirname(__file__), "..", "archsmith.json"), "r") as f:
        data = json.load(f)
    
    tracers: list[ASTracer] = []
    for key in data:
        tracers.append(ASTracer.from_dict(data[key]))
    
    tracers.sort(key=lambda obj: datetime.datetime.fromisoformat(obj.created).timestamp() if obj.created is not None else float('-inf'), reverse=True)
    return tracers

def main():
    print("ASVisual server is WIP.")