from flask import Flask, render_template, Response, jsonify
from detector.yolo_streamer import OptimizedYOLOStreaming
from pathlib import Path
import threading
import cv2
import os

app = Flask(__name__)
CAPTURA_DIR = Path("static/capturas")
CAPTURA_DIR.mkdir(parents=True, exist_ok=True)

streamer = OptimizedYOLOStreaming("rtsp://admin:123456@192.168.1.125:554/stream0")
streamer.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not streamer.frame_queue.empty():
                frame = streamer.frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ultimas')
def ultimas():
    imagenes = sorted(CAPTURA_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)[:3]
    return jsonify(imagenes=[f"/static/capturas/{img.name}" for img in imagenes])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
