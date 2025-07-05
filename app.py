



from ultralytics import YOLO
import cv2

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")

# URL de tu cámara IP (reemplaza con la real)
ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"

# Abrir el stream
cap = cv2.VideoCapture(ip_camera_url)
#cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)

vehiculos_interes = ['car', 'bus']

# Creamos una ventana única
cv2.namedWindow("Cámara IP - Vehículos", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede leer el stream")
        break

    # Detección
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in vehiculos_interes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar siempre en la misma ventana
    cv2.imshow("Cámara IP - Vehículos", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar
cap.release()
cv2.destroyAllWindows()
