import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("drone-detection-g4d3g").project("drone-detection-a1tsf")
dataset = project.version(6).download("yolov11", location="data/drones")
print("Dataset en:", dataset.location)
