from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")
ip_camera_url = "rtsp://admin:123456@192.168.1.125:554/stream0"
cap = cv2.VideoCapture(ip_camera_url)

vehiculos_interes = ['car', 'bus']
cv2.namedWindow("Cámara IP - Vehículos", cv2.WINDOW_NORMAL)

ultima_captura = 0  # Controla frecuencia de captura

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se puede leer el stream")
        break

    alto, ancho = frame.shape[:2]
    centro_frame = (ancho // 2, alto // 2)

    # Margen de tolerancia (10%)
    margen_x = int(ancho * 0.1)
    margen_y = int(alto * 0.2)

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

            # Dibujar recuadro en pantalla
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Si el centro está dentro del área central
            if (zona_central_x[0] <= cx <= zona_central_x[1]) and \
               (zona_central_y[0] <= cy <= zona_central_y[1]):

                tiempo_actual = time.time()
                if tiempo_actual - ultima_captura > 5:
                    # Recortar solo el vehículo detectado
                    auto_recortado = frame[y1:y2, x1:x2]

                    # Verifica que el recorte sea válido
                    if auto_recortado.size > 0:
                        nombre_archivo = f"vehiculo_centrado_{int(tiempo_actual)}.jpg"
                        cv2.imwrite(nombre_archivo, auto_recortado)
                        print(f"[✔] Imagen del vehículo guardada: {nombre_archivo}")
                        ultima_captura = tiempo_actual

    # Dibujar área central en el frame
    cv2.rectangle(frame,
                  (zona_central_x[0], zona_central_y[0]),
                  (zona_central_x[1], zona_central_y[1]),
                  (0, 255, 255), 2)

    cv2.imshow("Cámara IP - Vehículos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
