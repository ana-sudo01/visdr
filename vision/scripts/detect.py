import sys
import json
from ultralytics import YOLO

modelo =YOLO("runs/drones_v1/weights/best.pt")
imagen = sys.argv[1]

resultados = modelo(imagen, conf=0.25)
detecciones = []
