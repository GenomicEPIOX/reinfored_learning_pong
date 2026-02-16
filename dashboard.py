from flask import Flask, render_template, jsonify
import threading
import time

app = Flask(__name__)

# Shared metrics storage
metrics = {
    "episode": [],
    "score": [],
    "epsilon": [],
    "loss": [],
    "timestamp": []
}

def update_metrics(episode, score, epsilon, loss):
    metrics["episode"].append(episode)
    metrics["score"].append(score)
    metrics["epsilon"].append(epsilon)
    metrics["loss"].append(loss)
    metrics["timestamp"].append(time.time())

    # Keep history manageable (last 1000 episodes)
    if len(metrics["episode"]) > 1000:
        for key in metrics:
            metrics[key].pop(0)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

def run_dashboard(port=5000):
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_dashboard()
