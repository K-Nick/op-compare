import numpy as np
from types import SimpleNamespace
import dill

input_channel = 32
channel_multiplier = 2
h, w = 224, 224
kernel_size = 3
output_channel = 64

input_tensor = np.random.randn(1, h, w, input_channel) * 0.1
depthwise_filter = np.random.randn(kernel_size, kernel_size, input_channel, channel_multiplier) * 0.1
pointwise_filter = np.random.randn(1, 1, input_channel * channel_multiplier, output_channel) * 0.1

with open("./mock_data.pkl", "wb") as f:
    dill.dump(
        SimpleNamespace(input_tensor=input_tensor, depthwise_filter=depthwise_filter,
                        pointwise_filter=pointwise_filter), f)

print("Mock data generated and saved to ./mock_data.pkl")
