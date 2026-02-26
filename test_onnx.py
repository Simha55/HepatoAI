import onnxruntime as ort
sess = ort.InferenceSession("weights/yolo/yolo.onnx")
print(sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
print([(o.name, o.shape) for o in sess.get_outputs()])
