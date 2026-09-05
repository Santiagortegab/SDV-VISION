from ultralytics import YOLO
model_detect = YOLO("../../weights/yolo26n-seg.pt")  

resultados= model_detect.track(source="../../dashcam_example4.mp4", show=True, persist=True,  verbose=False)