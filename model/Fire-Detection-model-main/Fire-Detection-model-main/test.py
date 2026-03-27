from ultralytics import YOLO

model = YOLO ("runs/detect/train/weights/best.pt")

results = models("hello.mp4", save=True)