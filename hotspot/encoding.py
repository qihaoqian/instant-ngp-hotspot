import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder(encoding, encoding_config=None, **kwargs):
    if encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=3, num_levels=encoding_config.num_levels, level_dim=2, base_resolution=encoding_config.base_resolution, log2_hashmap_size=19,
                              desired_resolution=encoding_config.desired_resolution, gridtype='hash', align_corners=False)

    elif encoding == 'reg_grid':
        from hotspot.interpolators import RegularGridInterpolator
        encoder = RegularGridInterpolator(encoding_config)
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim