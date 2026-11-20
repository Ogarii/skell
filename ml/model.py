"""JointNet: a small U-Net that finds the 11 placed joints of a stick figure as heatmaps (PyTorch, training only).

Input: a framed drawing (synth.frame_drawing), size x size uint8, white paper. The net sees ink = 1 - img/255.
Output: one heatmap of logits per joint at 1/4 resolution (48 x 48 for 192 px). Cell (r, c) covers image pixels
[4c, 4c + 4) x [4r, 4r + 4), so its centre is at (4c + 2, 4r + 2) in the labels' pixel convention.

The same network runs in numpy in the sandbox (sandbox/joint_net.py) from export_numpy's weights, with each
batch norm folded into its convolution. tests/test_joint_net.py checks the two agree.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

WIDTHS = (32, 64, 128, 192, 256)
N_JOINTS = 11
STRIDE = 4                      # heatmap cell size in input pixels

# (block name, [(in, out, stride), ...]) for every conv-bn-relu stack, in the order the forward pass uses them.
# sandbox/joint_net.py walks the same layout.
def layout(w=WIDTHS):
    return [("stem", [(1, w[0], 2), (w[0], w[0], 1)]),                  # 96
            ("d1", [(w[0], w[1], 2), (w[1], w[1], 1)]),                  # 48  -> skip
            ("d2", [(w[1], w[2], 2), (w[2], w[2], 1)]),                  # 24  -> skip
            ("d3", [(w[2], w[3], 2), (w[3], w[3], 1)]),                  # 12  -> skip
            ("d4", [(w[3], w[4], 2), (w[4], w[4], 1), (w[4], w[4], 1)]),  # 6
            ("u3", [(w[4] + w[3], w[3], 1), (w[3], w[3], 1)]),           # 12
            ("u2", [(w[3] + w[2], w[2], 1), (w[2], w[2], 1)]),           # 24
            ("u1", [(w[2] + w[1], w[1], 1), (w[1], w[1], 1)])]           # 48


class CBR(nn.Sequential):
    def __init__(self, cin, cout, stride):
        super().__init__(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class JointNet(nn.Module):
    def __init__(self, widths=WIDTHS, n_joints=N_JOINTS):
        super().__init__()
        self.widths = tuple(widths)
        self.blocks = nn.ModuleDict({name: nn.Sequential(*[CBR(*s) for s in specs]) for name, specs in layout(widths)})
        self.head = nn.Conv2d(widths[1], n_joints, 1)

    def forward(self, ink):                    # ink (B, 1, H, W) in [0, 1]
        b = self.blocks
        x = b["stem"](ink)
        s1 = b["d1"](x)
        s2 = b["d2"](s1)
        s3 = b["d3"](s2)
        x = b["d4"](s3)
        x = b["u3"](torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s3], 1))
        x = b["u2"](torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s2], 1))
        x = b["u1"](torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), s1], 1))
        return self.head(x)                    # (B, J, H/4, W/4) logits


def to_ink(imgs_uint8: torch.Tensor) -> torch.Tensor:
    """(B, H, W) uint8 white paper -> (B, 1, H, W) float ink."""
    return (1.0 - imgs_uint8.float() / 255.0).unsqueeze(1)


def export_numpy(model: JointNet, path, extra: dict | None = None):
    """Save the weights for sandbox/joint_net.py: each conv with its batch norm folded in (eval statistics)."""
    model = model.eval().cpu()
    out = {"widths": np.array(model.widths), "stride": np.array(STRIDE)}
    for name, specs in layout(model.widths):
        for i, cbr in enumerate(model.blocks[name]):
            conv, bn = cbr[0], cbr[1]
            scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
            out[f"{name}.{i}.w"] = (conv.weight.detach() * scale[:, None, None, None]).numpy().astype(np.float32)
            out[f"{name}.{i}.b"] = (bn.bias.detach() - bn.running_mean * scale).numpy().astype(np.float32)
    out["head.w"] = model.head.weight.detach().numpy().astype(np.float32)
    out["head.b"] = model.head.bias.detach().numpy().astype(np.float32)
    for k, v in (extra or {}).items():
        out[k] = np.asarray(v)
    np.savez_compressed(path, **out)
