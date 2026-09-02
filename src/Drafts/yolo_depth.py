import cv2
import numpy as np
from ultralytics import YOLO

model_detect = YOLO("../../weights/yolo26n.pt")  
model_depth = YOLO("../../weights/yolo26n-depth.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resultados= model_detect.predict(source=frame, classes=[0,1,2,3], verbose=False)
    resultados_depth = model_depth.predict(source=frame, imgsz=768, verbose=False)
    mapa_profundidad = resultados_depth[0].depth.data.cpu().numpy().squeeze()

    for resultado in resultados:
        for caja in resultado.boxes:
            x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
            id_class = int(caja.cls[0].item())
            class_name = resultado.names[id_class]

            roi_profundidad = mapa_profundidad[y1:y2, x1:x2]
            
            if roi_profundidad.size > 0:
                distancia_z = np.min(roi_profundidad)
            else:
                distancia_z = 0.0


            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame,f"{class_name}: profundidad: {distancia_z:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

