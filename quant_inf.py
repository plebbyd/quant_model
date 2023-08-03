import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import numpy as np

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model_quant_8_full_edgetpu.tflite')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
rows = np.array([[0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8]
        ])

# Run an inference
common.set_input(interpreter, rows)
interpreter.invoke()
classes = classify.get_scores(interpreter)

print(f'classes: {classes}')
