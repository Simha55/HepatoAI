from ultralytics import YOLO
m = YOLO("weights/yolo/yolo.onnx")
print("names:", m.names)
print("num_names:", len(m.names))