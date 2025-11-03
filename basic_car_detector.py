# detect_cars.py
# Requiere: pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2
import time

SOURCE = 'rtsp://admin:9H)p5x84@192.168.1.64:554/Streaming/Channels/101'
CONFIDENCE = 0.35

def main():
    model = YOLO('yolov8n.pt')  # descarga automática la red pequeña (yolov8n)
    cap = cv2.VideoCapture(SOURCE)

    if not cap.isOpened():
        print("No se pudo abrir la fuente:", SOURCE)
        return

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inferencia: returns a list con un objeto Result para cada frame
        results = model.predict(frame, imgsz=640, conf=CONFIDENCE, verbose=False)

        # results puede contener varios frames; tomamos el primero
        r = results[0]
        boxes = getattr(r, "boxes", None)
        names = model.names  # diccionario id->nombre clase (p.ej. 'car')

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls)   # id de clase
                cls_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf)
                if cls_name == 'car' and conf >= CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) if hasattr(box, "xyxy") else map(int, box.xyxy)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # mostrar FPS
        dt = time.time() - fps_time
        fps = 1/dt if dt>0 else 0
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)

        cv2.imshow('Deteccion Autos', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
