import math
import torch
from torch import nn
from utilities.misc import NestedTensor

"""
Relative positional encoding with position embeddings
"""

class SinePositionalEncoding1D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        :param num_pos_feats: The number of features for the positional encoding (half the dimension).
        :param temperature: A scaling factor for the frequency of the sine waves.
        :param normalize: If True, scales the position by the `scale` parameter.
        :param scale: A scaling factor used when `normalize` is True. Defaults to 2*pi.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, inputs: NestedTensor):
        """
        Generates the sine positional encoding for the input tensor.

        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        x = inputs.left

        # update h and w if downsampling
        bs, _, h, w = x.size()
        if inputs.sampled_cols is not None:
            bs, w = inputs.sampled_cols.size()
        if inputs.sampled_rows is not None:
            _, h = inputs.sampled_rows.size()

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)

        # scale distance if there is down sample
        if inputs.sampled_cols is not None:
            scale = x.size(-1) / float(inputs.sampled_cols.size(-1))
            x_embed = x_embed * scale

        if self.normalize:
            x_embed = x_embed * self.scale

        # calculate positional encoding
        feature_dim = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        feature_dim = self.temperature ** (2 * (feature_dim // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / feature_dim  # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos


def no_pos_encoding(x):
    return None


def build_position_encoding(args):
    encoding_type = args.position_encoding
    channel_dim = args.channel_dim
    if encoding_type.lower() == 'sine1d_rel':
        n_features = channel_dim
        position_encoding = SinePositionalEncoding1D(n_features, normalize=False)
    elif encoding_type.lower() == 'none':
        position_encoding = no_pos_encoding
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    return position_encoding
