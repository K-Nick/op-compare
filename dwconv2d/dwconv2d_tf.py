import tensorflow as tf
import numpy as np
import dill

mock_data = dill.load(open("./mock_data.pkl", "rb"))

input_tensor = mock_data.input_tensor
depthwise_filter = mock_data.depthwise_filter
pointwise_filter = mock_data.pointwise_filter

output = tf.nn.separable_conv2d(input=input_tensor,
                                depthwise_filter=depthwise_filter,
                                pointwise_filter=pointwise_filter,
                                strides=[1, 1, 1, 1],
                                padding="SAME")
output = np.array(output)
with open("./output_tf.pkl", "wb") as f:
    dill.dump(output, f)

print("TF Output saved to ./output_tf.pkl")