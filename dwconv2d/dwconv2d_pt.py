import torch
import torch.nn as nn
from einops import rearrange, repeat
import dill
from types import SimpleNamespace


class SeparableConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, channel_multiplier=1, padding=0, bias=False) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels,
                                        channel_multiplier * in_channels,
                                        kernel_size,
                                        padding=padding,
                                        groups=in_channels,
                                        bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels * channel_multiplier, out_channels, 1, bias=bias)

    def forward(self, x):
        x0 = self.depthwise_conv(x)
        x1 = self.pointwise_conv(x0)
        return x0, x1

mock_data = dill.load(open("./mock_data.pkl", "rb"))
K, C_in, K_mult = mock_data.depthwise_filter.shape[1:]
C_out = mock_data.pointwise_filter.shape[-1]

input_tensor = rearrange(mock_data.input_tensor, "b h w c -> b c h w")
depthwise_filter = repeat(mock_data.depthwise_filter, "k k0 ci km -> (ci km) ax k k0", ax=1)
pointwise_filter = rearrange(mock_data.pointwise_filter, "k k0 ci co -> co ci k k0")

input_tensor = torch.from_numpy(input_tensor)
depthwise_filter = torch.from_numpy(depthwise_filter)
pointwise_filter = torch.from_numpy(pointwise_filter)

conv = SeparableConv2D(in_channels=C_in, out_channels=C_out, kernel_size=K, channel_multiplier=K_mult, padding="same")
print("conv.depthwise_conv.weight.shape", conv.depthwise_conv.weight.shape)
print("conv.pointwise_conv.weight.shape", conv.pointwise_conv.weight.shape)
conv.depthwise_conv.weight.data = depthwise_filter
conv.pointwise_conv.weight.data = pointwise_filter
print("conv.depthwise_conv.weight.shape", conv.depthwise_conv.weight.shape)
print("conv.pointwise_conv.weight.shape", conv.pointwise_conv.weight.shape)
after_dw, after_pw = conv(input_tensor)

after_dw = rearrange(after_dw, "b c h w -> b h w c")
after_dw = after_dw.detach().cpu().numpy()
after_pw = rearrange(after_pw, "b c h w -> b h w c")
after_pw = after_pw.detach().cpu().numpy()
with open("./output_pt.pkl", "wb") as f:
    dill.dump(after_pw, f)

print("PT Output saved to ./output_pt.pkl")