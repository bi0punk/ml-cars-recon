from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"
cap = cv2.VideoCapture(ip_camera_url)

vehiculos_interes = ['car', 'bus']
ultima_captura = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[!] No se puede leer el stream. Reintentando...")
        time.sleep(1)
        continue

    alto, ancho = frame.shape[:2]
    centro_frame = (ancho // 2, alto // 2)

    margen_x = int(ancho * 0.1)
    margen_y = int(alto * 0.1)

    zona_central_x = (centro_frame[0] - margen_x, centro_frame[0] + margen_x)
    zona_central_y = (centro_frame[1] - margen_y, centro_frame[1] + margen_y)

    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in vehiculos_interes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if (zona_central_x[0] <= cx <= zona_central_x[1]) and \
               (zona_central_y[0] <= cy <= zona_central_y[1]):

                tiempo_actual = time.time()
                if tiempo_actual - ultima_captura > 5:
                    auto_recortado = frame[y1:y2, x1:x2]
                    if auto_recortado.size > 0:
                        nombre_archivo = f"vehiculo_centrado_{int(tiempo_actual)}.jpg"
                        cv2.imwrite(nombre_archivo, auto_recortado)
                        print(f"[✔] Imagen del vehículo guardada: {nombre_archivo}")
                        ultima_captura = tiempo_actual

# Nunca se llama a cv2.imshow() ni cv2.waitKey()
cap.release()
