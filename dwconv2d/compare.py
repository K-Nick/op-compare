import dill
import numpy as np

with open("./output_pt.pkl", "rb") as f:
    output_pt = dill.load(f)

with open("./output_tf.pkl", "rb") as f:
    output_tf = dill.load(f)

print(np.allclose(output_pt, output_tf, atol=1e-6))