import cv2
import numpy as np
from ultralytics import YOLO

model_detect = YOLO("../../weights/yolo26n.pt")  
model_depth = YOLO("../../weights/yolo26n-depth.pt")

cap = cv2.VideoCapture("../../dashcam_example4.mp4")

historial_depth = {}
alpha = 0.2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    bev = np.ones((h, w, 3), dtype=np.uint8)*255
    cv2.rectangle(bev, (w//2-55, 0), (w//2+55, h), (230, 150, 50), -1)
    cv2.rectangle(bev, (w//2-30, h-60), (w//2+30, h), (0, 0, 0), -1)

    resultados= model_detect.track(source=frame, classes=[0,1,2,3], persist=True, verbose=False)
    resultados_depth = model_depth.predict(source=frame, imgsz=768, verbose=False)
    mapa_profundidad = resultados_depth[0].depth.data.cpu().numpy().squeeze()

    for resultado in resultados:

        for caja, id_tensor in zip(resultado.boxes, resultado.boxes.id):
            x1, y1, x2, y2 = caja.xyxy[0].int().tolist()
            xc = int((x1 + x2) / 2)
            yc = int((y1 + y2) / 2)
            id_class = int(caja.cls[0].item())
            id_item = int(id_tensor.item())
            class_name = resultado.names[id_class]

            roi_profundidad = mapa_profundidad[y1:y2, x1:x2]
            
            if roi_profundidad.size > 0:
                prof_actual = np.min(roi_profundidad)
            else:
                prof_actual = 0.0

            if id_item in historial_depth:
                profundidad = alpha * prof_actual + (1 - alpha) * historial_depth[id_item]
            else:
                profundidad = prof_actual

            historial_depth[id_item] = profundidad
            

            y_bev = int(h - (profundidad * 16))

            x_metros = (xc - w // 2) * profundidad / 400
            escala = 30
            x_real =  int(w // 2 + x_metros * escala)

            if class_name == "person":
                cv2.rectangle(bev, (x_real-15,y_bev-15), (x_real+15,y_bev+15), (0, 255, 0), -1)
            elif class_name == "bike":
                cv2.rectangle(bev, (x_real-20,y_bev-25), (x_real+20,y_bev+25), (0, 0, 255), -1)
            elif class_name == "car":
                cv2.rectangle(bev, (x_real-30,y_bev-30), (x_real+30,y_bev+30), (0, 0, 255), -1)
            

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame,f"{class_name}: profundidad: {profundidad:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    gui = np.hstack((frame, bev))
    cv2.imshow("GUI", gui)
    

    



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

