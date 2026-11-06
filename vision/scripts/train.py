import torch
from ultralytics import YOLO

dispositivo = 0 if torch.cuda.is_available() else "cpu"
print(f"entrenando en {dispositivo}")

model = YOLO("yolo11n.pt")
model.train(
    data="data/drones/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device=dispositivo,
    project="runs",
    name="drones_v1",
)