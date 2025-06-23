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
    return tracers

def preppedTracersForRendering():
    tracers = getTracers()
    data = [t.represent() for t in tracers]
    
    # Iterate through the tracer objects
    for i in range(len(data)):
        # Adjust created time to Singapore timezone (UTC+8)
        data[i]['created'] = (datetime.datetime.fromisoformat(data[i]['created']) + datetime.timedelta(minutes=480)).strftime("%d %B, %A, %Y %H:%M:%S%p") if data[i]['created'] else "N/A"
        
        if data[i]['started'] and isinstance(data[i]['started'], str) and data[i]['finished'] and isinstance(data[i]['finished'], str):
            # Compute the duration
            data[i]['duration'] = f"{(datetime.datetime.fromisoformat(data[i]['finished']) - datetime.datetime.fromisoformat(data[i]['started'])).total_seconds():.4f} seconds"
        else:
            # If started or finished is not available, set duration to "N/A"
            data[i]['duration'] = "N/A"
        
        # Adjust started and finished times to Singapore timezone (UTC+8)
        data[i]['started'] = (datetime.datetime.fromisoformat(data[i]['started']) + datetime.timedelta(minutes=480)).strftime("%d %B, %A, %Y %H:%M:%S%p") if data[i]['started'] else "N/A"
        data[i]['finished'] = (datetime.datetime.fromisoformat(data[i]['finished']) + datetime.timedelta(minutes=480)).strftime("%d %B, %A, %Y %H:%M:%S%p") if data[i]['finished'] else "N/A"
        
        # Iterate through report objects in the tracer
        for r_i in range(len(data[i]['reports'])):
            # If the report isn't the latest one, compute the time to when the next, more recent report was created
            if r_i > 0 and data[i]['reports'][r_i]['created'] and isinstance(data[i]['reports'][r_i]['created'], str) and data[i]['reports'][r_i-1]['created'] and isinstance(data[i]['reports'][r_i-1]['created'], str):
                data[i]['reports'][r_i]['timeToNext'] = f"{(datetime.datetime.fromisoformat(data[i]['reports'][r_i-1]['created']) - datetime.datetime.fromisoformat(data[i]['reports'][r_i]['created'])).total_seconds():.4f} seconds"
            else:
                data[i]['reports'][r_i]['timeToNext'] = "N/A"
        
        # Iterate through report objects in the tracer to adjust their created times
        for r_i in range(len(data[i]['reports'])):
            data[i]['reports'][r_i]['created'] = (datetime.datetime.fromisoformat(data[i]['reports'][r_i]['created']) + datetime.timedelta(minutes=480)).strftime("%d %B, %A, %Y %H:%M:%S%p") if data[i]['reports'][r_i]['created'] else "N/A"
    
    outputData = {}
    for tracer in data:
        # Use the tracer's ID as the key
        outputData[tracer['id']] = tracer
    
    return outputData

@app.route('/')
def index():
    return render_template('index.html', tracerData=preppedTracersForRendering(), lastUpdate=datetime.datetime.now().strftime("%d %B, %A, %Y %H:%M:%S%p"))

@app.route('/tracer/<tracer_id>')
def tracerDetail(tracer_id):
    tracers = preppedTracersForRendering()
    if tracer_id not in tracers:
        return jsonify({"error": "Tracer not found"}), 404
    tracer = tracers[tracer_id]
    
    return render_template('run.html', tracerInfo=tracer, lastUpdate=datetime.datetime.now().strftime("%d %B, %A, %Y %H:%M:%S%p"))

def main():
    app.run(host='0.0.0.0', port=8001)

if __name__ == '__main__':
    main()