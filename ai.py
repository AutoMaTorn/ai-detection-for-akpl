from ultralytics import YOLO
import torch
model = YOLO('yolov8n.pt')


results = model(source=0, show=True, conf=0.4, stream=True)
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
