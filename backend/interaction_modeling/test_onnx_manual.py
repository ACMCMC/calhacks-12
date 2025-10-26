import onnxruntime as ort
import numpy as np

# Manually specified input features
features = np.array([
    33.71000003814697, 0.10078271025809173, 143.04800009727478, 3.8833333333333333, 2.8666666666666667, 0.075, 0, 0, 0, 0, 0, 0.5156794425087095, 7.141666666666667, 1, 0.009822055892516618, 75, 0, 0.7347675306409499, 7.266565481660445, 0.6513479411838575
], dtype=np.float32).reshape(1, -1)

onnx_path = '../../models/interaction_predictor.onnx'
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name
print('ONNX input name:', input_name)

outputs = session.run(None, {input_name: features})
print('ONNX raw outputs:', outputs)

# Try to print probabilities if present
for out in outputs:
    if isinstance(out, np.ndarray) and out.shape[-1] == 2:
        print('Probabilities:', out)
        print('Positive class probability:', out[0,1])
