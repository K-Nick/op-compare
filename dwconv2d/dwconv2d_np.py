import numpy as np
import dill
from itertools import product
from einops import einsum

mock_data = dill.load(open("./mock_data.pkl", "rb"))
input_tensor = mock_data.input_tensor[0]
depthwise_filter = mock_data.depthwise_filter
pointwise_filter = mock_data.pointwise_filter

padded_tensor = np.pad(input_tensor, ((1, 1), (1, 1), (0, 0)), mode="constant")
H, W = input_tensor.shape[:2]
H_, W_, Cin = padded_tensor.shape
K = depthwise_filter.shape[0]
Km = depthwise_filter.shape[-1]
Cout = pointwise_filter.shape[-1]

output_tensor_dw = np.zeros((H, W, Cin*Km))
for i, j in product(range(H), range(W)):
    output_tensor_dw[i, j, :] = einsum(padded_tensor[i:i + 3, j:j + 3], depthwise_filter, "kh kw ci, kh kw ci km -> ci km").flatten()

output_tensor_pw = np.zeros((H, W, Cout))
for i,j in product(range(H), range(W)):
    output_tensor_pw[i, j, :] = einsum(output_tensor_dw[i, j, :], pointwise_filter[0,0], "ci, ci co -> co").flatten()

import ipdb
ipdb.set_trace() #FIXME ipdb