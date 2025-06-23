import os, json, datetime
from flask import Flask, render_template, request, jsonify
from addons import ASTracer

app = Flask(__name__)

def getTracers():
    with open(os.path.join(os.path.dirname(__file__), "..", "archsmith.json"), "r") as f:
        data = json.load(f)
    
    tracers: list[ASTracer] = []
    for key in data:
        tracers.append(ASTracer.from_dict(data[key]))
    
    tracers.sort(key=lambda obj: datetime.datetime.fromisoformat(obj.created).timestamp() if obj.created is not None else float('-inf'), reverse=True)
    for tracer in tracers:
        tracer.reports.sort(key=lambda obj: datetime.datetime.fromisoformat(obj.created).timestamp() if obj.created is not None else float('-inf'), reverse=True)
        for report in tracer.reports:
            report.created = datetime.datetime.fromisoformat(report.created).strftime("%d %B, %A, %Y %H:%M:%S%p") if report.created else "N/A"
        
        tracer.created = datetime.datetime.fromisoformat(tracer.created).strftime("%d %B, %A, %Y %H:%M:%S%p") if tracer.created else "N/A"
        tracer.started = datetime.datetime.fromisoformat(tracer.started).strftime("%d %B, %A, %Y %H:%M:%S%p") if tracer.started else "N/A"
        tracer.finished = datetime.datetime.fromisoformat(tracer.finished).strftime("%d %B, %A, %Y %H:%M:%S%p") if tracer.finished else "N/A"
    return tracers

@app.route('/')
def index():
    return render_template('index.html', tracerData={t.id: t.represent() for t in getTracers()})

def main():
    app.run(host='0.0.0.0', port=8001)