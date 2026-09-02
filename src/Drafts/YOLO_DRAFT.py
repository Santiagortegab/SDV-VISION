from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")  

results = model.predict(source="0", show=True, save=True)
