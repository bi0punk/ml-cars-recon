from flask import Flask, Response, render_template, jsonify, send_from_directory
import threading
import cv2
import time
from yolo_004 import OptimizedYOLOStreaming

app = Flask(__name__)
streamer = None

# Endpoint para video MJPEG
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if streamer and streamer.frame_web is not None:
                ret, jpeg = cv2.imencode('.jpg', streamer.frame_web)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.05)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API para obtener últimas imágenes
@app.route('/api/last_images')
def last_images():
    if streamer:
        with streamer.lock:
            return jsonify(streamer.last_captured_images)
    return jsonify([])

# Servir imágenes estáticas
@app.route('/captures/<path:filename>')
def serve_capture(filename):
    return send_from_directory('capturas', filename)

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"
    streamer = OptimizedYOLOStreaming(ip_camera_url)
    
    # Iniciar procesamiento en hilo separado
    processing_thread = threading.Thread(target=streamer.run)
    processing_thread.daemon = True
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000, threaded=True)