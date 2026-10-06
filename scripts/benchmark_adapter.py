"""Full PENet_C2, xyz geometry, with explicit NYU-only input adaptation."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAME = 'PENet'
DCN_BACKEND = 'none (CSPN unfold/einsum propagation)'


def model_config(seed, data_dir):
    return SimpleNamespace(seed=seed, from_scratch=True, prop_time=12,
                           network_model='pe', variant='PENet_C2', dilation_rate=2,
                           convolutional_layer_encoding='xyz', geometry_size=[256, 320],
                           input_hw=[228, 304], internal_hw=[256, 320],
                           padding_lrtb=[0, 16, 0, 28], padding_mode='constant zero',
                           propagation='6 iterations at dilation 2, then 6 at dilation 1; kernels 3/5/7',
                           camera='NYU RGB intrinsics, resize 0.5 and center crop (left=8, top=6)',
                           rgb_preprocessing='common ImageNet-normalized NYU input, not original KITTI uint8',
                           limitation='NYU shape adaptation; not a trained NYU or original KITTI reproduction')


def build_model(args):
    from model import PENet_C2
    return PENet_C2(args)


class DepthOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        # Use the same endpoint-normalized coordinate convention as CoordConv.
        height, width = net.backbone.args.geometry_size
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing='ij')
        position = torch.stack((2 * x / (width - 1) - 1,
                                2 * y / (height - 1) - 1), dim=0).unsqueeze(0)
        intrinsic = torch.tensor([[582.62448167737955 / 2, 0, 313.04475870804731 / 2 - 8],
                                  [0, 582.69103270988637 / 2, 238.44389626620386 / 2 - 6],
                                  [0, 0, 1]], dtype=torch.float32).unsqueeze(0)
        self.register_buffer('position', position, persistent=False)
        self.register_buffer('K', intrinsic, persistent=False)

    def forward(self, rgb, dep):
        height, width = dep.shape[-2:]
        padding = self.net.backbone.args.padding_lrtb
        result = self.net({'rgb': F.pad(rgb, padding), 'd': F.pad(dep, padding),
                           'position': self.position, 'K': self.K})
        return result[:, :, :height, :width]


def reference_prediction(model, rgb, dep):
    # Invoke the complete original forward on the same explicitly adapted input.
    padding = model.net.backbone.args.padding_lrtb
    sample = {'rgb': F.pad(rgb, padding), 'd': F.pad(dep, padding),
              'position': model.position, 'K': model.K}
    return model.net(sample)[:, :, :dep.shape[-2], :dep.shape[-1]]
