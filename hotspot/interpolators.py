import math
from dataclasses import dataclass
from itertools import product
from typing import Union

import torch
import torch.nn as nn

from config.config import ConfigABC
# from erl_neural_sddf.models.base import ModuleBase

__all__ = [
    "RegularGridInterpolator",
    "CircularGridInterpolator",
    "SphericalGridInterpolator",
    "PositionDirectionInterpolator",
]


class RegularGridInterpolator(nn.Module):
    """
    Adapted for multivariate data from https://github.com/sbarratt/torch_interpolations
    Differentiable grid interpolator.
    """

    @dataclass
    class Config(ConfigABC):
        feature_dim: int = 32
        grid_dim: int = 3
        grid_min: tuple[float] = (-1, -1, -1)
        grid_max: tuple[float] = (1, 1, 1)
        grid_res: tuple[float] = (0.05,0.05,0.05)  # resolution of the grid

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert self.config.grid_dim == len(self.config.grid_min)
        assert self.config.grid_dim == len(self.config.grid_max)
        assert self.config.grid_dim == len(self.config.grid_res)

        
        self.output_dim = self.config.feature_dim

        self.grid_shape = [self.config.feature_dim]  # 直接用 int 包成 list

        for i, (lb, ub, res) in enumerate(zip(self.config.grid_min, self.config.grid_max, self.config.grid_res)):
            assert lb < ub, f"grid_min[{i}] must be less than grid_max[{i}]"
            assert res > 0, f"grid_res[{i}] must be greater than 0"
            ticks = torch.arange(lb, ub, res)  # [lb, ub) with res
            self.grid_shape.append(ticks.size(0))
            self.register_buffer(f"grid_ticks_{i}", ticks)
        self.grid_shape = tuple(self.grid_shape)

        param = nn.Parameter(torch.rand(self.grid_shape, dtype=torch.float32))  # m x D1 x D2 x ... x DN tensor
        nn.init.xavier_normal_(param)  # this makes the network better
        self.register_parameter("grid_values", param)

    def forward(self, points_to_interp: torch.Tensor, feature_slice: Union[slice, list] = slice(None, None, None)):
        """
        Args:
            points_to_interp: n x N tensor of points to interpolate where n is the number of dimensions and N is the
                number of points to interpolate
            feature_slice: slice object to slice the feature dimension
        Returns:
            n x N tensor of local coordinates and m x N tensor of interpolated grid values
        """
        assert points_to_interp.size(1) == self.config.grid_dim
        points_to_interp = points_to_interp.transpose(0, 1)  # N x n tensor, where n is the number of dimensions
        points_to_interp = points_to_interp.contiguous()

        if feature_slice is None:
            feature_slice = [slice(None, None, None)]
        elif isinstance(feature_slice, slice):
            feature_slice = [feature_slice]
        assert isinstance(feature_slice, list)

        idxs = []
        dists = []
        overalls = []
        local_coords = []  # local coordinates of the query points in the corresponding grid cell
        # iterate over each dimension
        for dim_idx, x in enumerate(points_to_interp):  # ticks are the boundaries and x are the query point
            ticks = getattr(self, f"grid_ticks_{dim_idx}")  # get the tick marks along this dimension
            idx_right = torch.bucketize(x, ticks)  # find the index of the right boundary
            idx_right[idx_right >= ticks.shape[0]] = ticks.shape[0] - 1  # clamp the index to the last element
            idx_left = (idx_right - 1).clamp(0, ticks.shape[0] - 1)  # find the index of the left boundary
            dist_left = x - ticks[idx_left]  # distance from the left boundary
            dist_right = ticks[idx_right] - x  # distance from the right boundary

            # assert not torch.any(dist_left < 0), "some points to interpolate are out of boundary."
            # assert not torch.any(dist_right < 0), "some points to interpolate are out of boundary."

            dist_left[dist_left < 0] = 0.0  # clamp the distance to 0 if it is negative
            dist_right[dist_right < 0] = 0.0  # clamp the distance to 0 if it is negative

            both_zero = (dist_left == 0) & (dist_right == 0)  # find points on the left boundary
            dist_left[both_zero] = dist_right[both_zero] = 1.0  # these points are interpolated without this dimension

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)
            local_coords.append(dist_left / overalls[-1])

        numerator = 0.0
        thing = list(product([0, 1], repeat=points_to_interp.size(0)))
        for indexer in thing:  # iterate over all possible combinations of left and right boundaries
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]  # value index
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]  # weight
            mp = torch.prod(torch.stack(bs_s), dim=0)

            inds = feature_slice + as_s  # self.grid_values has shape [..., dim1, dim2, ..., dimN]
            numerator += self.grid_values[inds] * mp
        denominator = torch.prod(torch.stack(overalls), dim=0)
        features = numerator / denominator
        local_coords = torch.stack(local_coords)
        # return local_coords, features
        return features.transpose(0,1)

    def resample(self, factor: Union[float, list[float]]):
        if isinstance(factor, float):
            factor = [factor] * self.config.grid_dim
        assert len(factor) == self.config.grid_dim, "factor must be a float or a list of floats with length grid_dim"
        new_config = RegularGridInterpolator.Config()
        new_config.feature_dim = self.config.feature_dim
        new_config.grid_dim = self.config.grid_dim
        new_config.grid_min = self.config.grid_min
        new_config.grid_max = self.config.grid_max
        new_config.grid_res = (res * f for res, f in zip(self.config.grid_res, factor))
        new_features = RegularGridInterpolator(new_config).to(self.cur_device)
        grid_points = torch.meshgrid(  # n D1 x D2 x ... x Dn tensors
            [getattr(new_features, f"grid_ticks_{i}") for i in range(new_config.grid_dim)],
            indexing="ij",
        )
        grid_points = [p.flatten() for p in grid_points]  # n N-dim tensors, N = D1 x D2 x ... x Dn
        grid_points = torch.stack(grid_points, dim=0).to(self.cur_device)  # n x N tensor
        interpolated_grid_values = self.forward(grid_points)  # m x N tensor
        new_features.grid_values.data[...] = interpolated_grid_values.view(new_features.grid_shape)
        return new_features


def slerp(
    f1: torch.Tensor,
    f2: torch.Tensor,
    theta: torch.Tensor,
    omega: float,
    normalized: bool,
) -> torch.Tensor:
    w1 = torch.sin(omega - theta)
    w2 = torch.sin(theta)
    if normalized:
        s = w1 + w2
        w1 = w1 / s
        w2 = w2 / s
    else:
        so = math.sin(omega)
        w1 = w1 / so
        w2 = w2 / so
    return w1 * f1 + w2 * f2


def slerp_so2(
    angles: torch.Tensor,
    grid_values: torch.Tensor,
    grid_ticks: torch.Tensor,
    normalized: bool,
) -> torch.Tensor:
    assert grid_ticks.ndim == 1, "grid_ticks must be 1D"

    num_angles = grid_ticks.numel()
    pi = torch.pi

    assert torch.all(angles >= -pi) and torch.all(angles < pi), "angles must be in [-pi, pi)"

    idx_right = torch.bucketize(angles, grid_ticks)
    idx_left = idx_right - 1
    idx_right[idx_right >= num_angles] = 0  # wrap around

    thetas = angles - grid_ticks[idx_left]
    omega = grid_ticks[1] - grid_ticks[0]
    f = slerp(grid_values[..., idx_left], grid_values[..., idx_right], thetas, omega, normalized)
    return f


def slerp_so3(
    azimuths: torch.Tensor,
    elevations: torch.Tensor,
    grid_values: torch.Tensor,
    grid_values_poles: torch.Tensor,
    grid_ticks_azimuth: torch.Tensor,
    grid_ticks_elevation: torch.Tensor,
    normalized: bool,
) -> torch.Tensor:
    assert grid_ticks_azimuth.ndim == 1, "grid_ticks_azimuth must be 1D"
    assert grid_ticks_elevation.ndim == 1, "grid_ticks_elevation must be 1D"

    num_azimuths = grid_ticks_azimuth.numel()
    num_elevations = grid_ticks_elevation.numel() + 2  # include the two poles
    pi_2 = torch.pi / 2

    assert torch.all((azimuths >= -torch.pi) & (azimuths < torch.pi)), "azimuth must be in [-pi, pi)."
    assert torch.all((elevations >= -pi_2) & (elevations <= pi_2)), "elevation must be in [-pi/2, pi/2]."

    azimuth_idx_right = torch.bucketize(azimuths, grid_ticks_azimuth)
    azimuth_idx_left = azimuth_idx_right - 1
    azimuth_idx_right[azimuth_idx_right >= num_azimuths] = 0  # wrap around
    # azimuth_idx_left[azimuth_idx_left < 0] = self.config.num_azimuths - 1  # not needed
    elevation_idx_right = torch.bucketize(elevations, grid_ticks_elevation)

    if num_elevations == 3:
        # slerp between the two azimuths
        thetas = azimuths - grid_ticks_azimuth[azimuth_idx_left]
        omega = grid_ticks_azimuth[1] - grid_ticks_azimuth[0]
        f = slerp(
            grid_values[..., azimuth_idx_left, 0],
            grid_values[..., azimuth_idx_right, 0],
            thetas,
            omega,
            normalized,
        )

        # slerp between the two poles
        thetas = torch.abs(elevations)
        # negative elevation -> pole 0, positive elevation -> pole 1
        ff = slerp(f, grid_values_poles[..., elevation_idx_right], thetas, pi_2, normalized)
        return ff
    else:
        # self.grid_ticks_elevation has shape (ne, )
        # if elevation_idx_right == 0, then the elevation is the south pole
        # if elevation_idx_right == ne + 1, then the elevation is the north pole
        # otherwise, the elevation is in the middle

        mask_south_pole = elevation_idx_right == 0
        mask_north_pole = elevation_idx_right == num_elevations - 2
        mask_middle = ~mask_south_pole & ~mask_north_pole

        feature_dim = list(grid_values.shape)[:-2]  # remove the last two dimensions (azimuth, elevation)
        ff = torch.empty((*feature_dim, azimuths.size(0)), device=grid_values.device)  # (f1, f2, ..., N)
        omega_azimuth = grid_ticks_azimuth[1] - grid_ticks_azimuth[0]
        omega_elevation = grid_ticks_elevation[0] + pi_2

        # deal with the points in the middle
        # slerp between the two azimuths for the middle elevations
        azimuth_idx_left_middle = azimuth_idx_left[mask_middle]
        if len(azimuth_idx_left_middle) > 0:
            azimuth_idx_right_middle = azimuth_idx_right[mask_middle]
            elevation_idx_right_middle = elevation_idx_right[mask_middle]
            elevation_idx_left_middle = elevation_idx_right_middle - 1
            thetas = azimuths[mask_middle] - grid_ticks_azimuth[azimuth_idx_left_middle]
            assert torch.all(thetas >= 0)
            f_bottom = slerp(
                grid_values[..., azimuth_idx_left_middle, elevation_idx_left_middle],
                grid_values[..., azimuth_idx_right_middle, elevation_idx_left_middle],
                thetas,
                omega_azimuth,
                normalized,
            )
            f_top = slerp(
                grid_values[..., azimuth_idx_left_middle, elevation_idx_right_middle],
                grid_values[..., azimuth_idx_right_middle, elevation_idx_right_middle],
                thetas,
                omega_azimuth,
                normalized,
            )
            # slerp between the two elevations
            thetas = elevations[mask_middle] - grid_ticks_elevation[elevation_idx_left_middle]
            assert torch.all(thetas >= 0)
            ff[..., mask_middle] = slerp(f_bottom, f_top, thetas, omega_elevation, normalized)

        # deal with the points at the south pole
        # slerp between the two azimuths for the south pole
        azimuth_idx_left_sp = azimuth_idx_left[mask_south_pole]
        if len(azimuth_idx_left_sp) > 0:
            azimuth_idx_right_sp = azimuth_idx_right[mask_south_pole]
            thetas = azimuths[mask_south_pole] - grid_ticks_azimuth[azimuth_idx_left_sp]
            assert torch.all(thetas >= 0)
            f = slerp(
                grid_values[..., azimuth_idx_left_sp, 0],
                grid_values[..., azimuth_idx_right_sp, 0],
                thetas,
                omega_azimuth,
                normalized,
            )

            thetas = elevations[mask_south_pole] + pi_2
            ff[..., mask_south_pole] = slerp(grid_values_poles[:, [0]], f, thetas, omega_elevation, normalized)

        # deal with the points at the north pole
        # slerp between the two azimuths for the north pole
        azimuth_idx_left_np = azimuth_idx_left[mask_north_pole]
        if len(azimuth_idx_left_np) > 0:
            azimuth_idx_right_np = azimuth_idx_right[mask_north_pole]
            thetas = azimuths[mask_north_pole] - grid_ticks_azimuth[azimuth_idx_left_np]
            assert torch.all(thetas >= 0)
            f = slerp(
                grid_values[..., azimuth_idx_left_np, -1],
                grid_values[..., azimuth_idx_right_np, -1],
                thetas,
                omega_azimuth,
                normalized,
            )

            thetas = pi_2 - elevations[mask_north_pole]
            ff[..., mask_north_pole] = slerp(grid_values_poles[:, [1]], f, thetas, omega_elevation, normalized)

        return ff


class CircularGridInterpolator(nn.Module):
    @dataclass
    class Config(ConfigABC):
        feature_dim: Union[tuple, int] = 32
        num_angles: int = 4
        normalized: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.config: CircularGridInterpolator.Config
        assert self.config.num_angles >= 2, "num_angles must be >= 2"

        if isinstance(self.config.feature_dim, int):
            self.config.feature_dim = (self.config.feature_dim,)  # make it a tuple

        grid_values = nn.Parameter(torch.rand((*self.config.feature_dim, self.config.num_angles), dtype=torch.float32))
        nn.init.xavier_normal_(grid_values)
        self.register_parameter("grid_values", grid_values)

        grid_ticks = torch.linspace(-torch.pi, torch.pi, self.config.num_angles + 1)[:-1]  # exclude the last point
        self.register_buffer("grid_ticks", grid_ticks)

    def forward(self, points_to_interp: torch.Tensor):
        """
        Args:
            points_to_interp: (N, ) tensor of angles or (2, N) tensor of x, y
        Returns:
            m x N tensor of interpolated grid values, where m is the feature dimension
        """
        if points_to_interp.ndim == 2:
            assert points_to_interp.size(0) == 2, "each column of points_to_interp must be (x, y)"
            points_to_interp = torch.atan2(points_to_interp[1], points_to_interp[0])
        elif points_to_interp.ndim != 1:
            raise ValueError("points_to_interp must be a 1D tensor of angles or a 2D tensor of x, y")

        f = slerp_so2(points_to_interp, self.grid_values, self.grid_ticks, self.config.normalized)
        return f


class SphericalGridInterpolator(nn.Module):
    @dataclass
    class Config(ConfigABC):
        feature_dim: Union[tuple, int] = 32
        num_azimuths: int = 4
        num_elevations: int = 3
        normalized: bool = False  # if True, the interpolation output is bounded by its corresponding grid values

    def __init__(self, config: Config):
        super().__init__(config)
        self.config: SphericalGridInterpolator.Config

        assert self.config.num_azimuths >= 2, "num_azimuths must be >= 2"
        assert self.config.num_elevations >= 3, "num_elevations must be >= 3"

        if isinstance(self.config.feature_dim, int):
            self.config.feature_dim = (self.config.feature_dim,)  # make it a tuple

        na = self.config.num_azimuths
        ne = self.config.num_elevations - 2  # exclude two poles
        grid_values = nn.Parameter(torch.rand((*self.config.feature_dim, na, ne), dtype=torch.float32))
        nn.init.xavier_normal_(grid_values)
        self.register_parameter("grid_values", grid_values)
        grid_values_poles = nn.Parameter(torch.rand((*self.config.feature_dim, 2), dtype=torch.float32))  # two poles
        nn.init.xavier_normal_(grid_values_poles)
        self.register_parameter("grid_values_poles", grid_values_poles)
        # torch.linspace includes both ends
        pi_2 = torch.pi / 2
        grid_ticks_azimuth = torch.linspace(-torch.pi, torch.pi, na + 1)[:-1]  # exclude the last point
        grid_ticks_elevation = torch.linspace(-pi_2, pi_2, ne + 2)[1:-1]  # exclude the first and last
        self.register_buffer("grid_ticks_azimuth", grid_ticks_azimuth)
        self.register_buffer("grid_ticks_elevation", grid_ticks_elevation)

    def forward(self, points_to_interp: torch.Tensor):
        """
        Args:
            points_to_interp: 2 x N tensor of azimuth and elevation, or 3 x N tensor of x, y, z
        Returns:
            m x N tensor of interpolated grid values, where m is the feature dimension
        """
        if points_to_interp.size(0) == 3:
            azimuths = torch.atan2(points_to_interp[1], points_to_interp[0])
            elevations = torch.asin(points_to_interp[2])
        elif points_to_interp.size(0) == 2:
            azimuths = points_to_interp[0].contiguous()
            elevations = points_to_interp[1].contiguous()
        else:
            raise ValueError("each column of points_to_interp must be (azimuth, elevation) or (x, y, z)")

        ff = slerp_so3(
            azimuths,
            elevations,
            self.grid_values,
            self.grid_values_poles,
            self.grid_ticks_azimuth,
            self.grid_ticks_elevation,
            self.config.normalized,
        )
        return ff


class PositionDirectionInterpolator(nn.Module):
    @dataclass
    class Config(ConfigABC):
        feature_dim: Union[tuple, int] = 32
        space_dim: int = 2
        position_min: tuple = (0, 0)
        position_max: tuple = (100, 100)
        position_res: tuple = (5, 5)
        direction_grid_shape: tuple = (4,)
        normalized: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.config: PositionDirectionInterpolator.Config
        assert self.config.space_dim == len(self.config.position_min)
        assert self.config.space_dim == len(self.config.position_max)
        assert self.config.space_dim == len(self.config.position_res)
        assert self.config.space_dim - 1 == len(self.config.direction_grid_shape)

        if isinstance(self.config.feature_dim, int):
            self.config.feature_dim = (self.config.feature_dim,)  # make it a tuple

        # both position and direction interpolators are linear interpolators
        # the order of calling them does not matter
        # considering performance, it is better to do the direction interpolation first
        if self.config.space_dim == 2:
            assert self.config.direction_grid_shape[0] >= 2, "num_angles must be >= 2"
            pos_interpolator_config = RegularGridInterpolator.Config(
                feature_dim=(*self.config.feature_dim, self.config.direction_grid_shape[0]),
                grid_dim=2,
                grid_min=self.config.position_min,
                grid_max=self.config.position_max,
                grid_res=self.config.position_res,
            )
            self.position_interpolator = RegularGridInterpolator(pos_interpolator_config)
            grid_ticks_azimuth = torch.linspace(-torch.pi, torch.pi, self.config.direction_grid_shape[0] + 1)[:-1]
            self.register_buffer("grid_ticks_azimuth", grid_ticks_azimuth)
        elif self.config.space_dim == 3:
            assert self.config.direction_grid_shape[0] >= 2, "num_azimuths must be >= 2"
            assert self.config.direction_grid_shape[1] >= 3, "num_elevations must be >= 3"
            pos_interpolator_config = RegularGridInterpolator.Config(
                feature_dim=(
                    *self.config.feature_dim,
                    self.config.direction_grid_shape[0],
                    self.config.direction_grid_shape[1] - 2,
                ),
                grid_dim=3,
                grid_min=self.config.position_min,
                grid_max=self.config.position_max,
                grid_res=self.config.position_res,
            )
            self.position_interpolator = RegularGridInterpolator(pos_interpolator_config)
            pole_interpolator_config = RegularGridInterpolator.Config(
                feature_dim=(*self.config.feature_dim, 2),
                grid_dim=3,
                grid_min=self.config.position_min,
                grid_max=self.config.position_max,
                grid_res=self.config.position_res,
            )
            self.pole_interpolator = RegularGridInterpolator(pole_interpolator_config)
            pi_2 = torch.pi / 2
            grid_ticks_azimuth = torch.linspace(-torch.pi, torch.pi, self.config.direction_grid_shape[0] + 1)[:-1]
            grid_ticks_elevation = torch.linspace(-pi_2, pi_2, self.config.direction_grid_shape[1])[1:-1]
            self.register_buffer("grid_ticks_azimuth", grid_ticks_azimuth)
            self.register_buffer("grid_ticks_elevation", grid_ticks_elevation)
        else:
            raise ValueError("space_dim must be 2 or 3")

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """
        Args:
            positions: 2 x N tensor of positions, or 3 x N tensor of x, y, z
            directions: when space_dim is 2, directions is a 1D tensor of angles or a 2 x N tensor of x, y
                when space_dim is 3, directions is a 2 x N tensor of azimuth, elevation or a 3 x N tensor of x, y, z
        Returns:
            local_coords: dictionary of local coordinates of the query points in the corresponding grid cell
            f: m x N tensor of interpolated grid values, where m is the feature dimension
        """
        if self.config.space_dim == 2:
            if directions.ndim == 2:
                assert directions.size(0) == 2, "each column of directions must be (x, y)"
                directions = torch.atan2(directions[1], directions[0])
            elif directions.ndim != 1:
                raise ValueError("directions must be a 1D tensor of angles or a 2D tensor of x, y")
            assert positions.size(1) == directions.size(0)
            # compute feature_slice
            idx_right = torch.bucketize(directions, self.grid_ticks_azimuth)
            idx_left = idx_right - 1
            idx_right[idx_right >= self.config.direction_grid_shape[0]] = 0  # wrap around
            idx_left = idx_left.detach().cpu().numpy().tolist()
            idx_right = idx_right.detach().cpu().numpy().tolist()
            feature_slice = [slice(None, None, None), [idx_left, idx_right]]
            # interpolate
            local_positions, grid_values = self.position_interpolator(
                positions, feature_slice
            )  # f1 x ... x 2 x N tensor
            omega = self.grid_ticks_azimuth[1] - self.grid_ticks_azimuth[0]
            thetas = directions - self.grid_ticks_azimuth[idx_left]
            f = slerp(grid_values[..., 0, :], grid_values[..., 1, :], thetas, omega, self.config.normalized)
            return f
        else:
            pi = torch.pi
            pi_2 = torch.pi / 2
            if directions.size(0) == 3:
                azimuths = torch.atan2(directions[1], directions[0])
                elevations = torch.asin(directions[2])
            elif directions.size(0) == 2:
                azimuths = directions[0].contiguous()
                elevations = directions[1].contiguous()
                assert torch.all((azimuths >= -pi) & (azimuths < pi)), "azimuth must be in [-pi, pi)."
                assert torch.all((elevations >= -pi_2) & (elevations <= pi_2)), "elevation must be in [-pi/2, pi/2]."
            else:
                raise ValueError("each column of directions must be (azimuth, elevation) or (x, y, z)")
            assert positions.size(1) == azimuths.size(0)
            assert positions.size(1) == elevations.size(0)

            num_azimuths = self.config.direction_grid_shape[0]
            num_elevations = self.config.direction_grid_shape[1]
            azimuth_idx_right = torch.bucketize(azimuths, self.grid_ticks_azimuth)
            azimuth_idx_left = azimuth_idx_right - 1
            azimuth_idx_right[azimuth_idx_right >= num_azimuths] = 0
            if num_elevations == 3:
                # slerp between the two azimuths
                # compute feature_slice
                azimuth_idx_left = azimuth_idx_left.detach().cpu().numpy().tolist()
                azimuth_idx_right = azimuth_idx_right.detach().cpu().numpy().tolist()
                feature_slice = [slice(None, None, None), [azimuth_idx_left, azimuth_idx_right], 0]
                # interpolate
                local_positions, grid_values = self.position_interpolator(positions, feature_slice)
                local_azimuths = azimuths - self.grid_ticks_azimuth[azimuth_idx_left]
                omega = self.grid_ticks_azimuth[1] - self.grid_ticks_azimuth[0]
                f = slerp(  # grid_values has shape (f1, f2, ..., 2, N)
                    grid_values[..., 0, :],
                    grid_values[..., 1, :],
                    local_azimuths,
                    omega,
                    self.config.normalized,
                )  # f has shape (f1, f2, ..., N)
                local_azimuths /= omega
                # slerp between the two elevations
                # compute feature_slice
                elevation_idx_right = torch.bucketize(elevations, self.grid_ticks_elevation)
                feature_slice = [slice(None, None, None), elevation_idx_right.detach().cpu().numpy().tolist()]
                # interpolate
                local_positions, grid_values = self.pole_interpolator(positions, feature_slice)
                local_elevations = torch.abs(elevations)
                # grid_values has shape (f1, f2, ..., N)
                ff = slerp(f, grid_values, local_elevations, pi_2, self.config.normalized)
                local_elevations /= pi_2

                local_coords = dict(
                    positions=local_positions,
                    azimuths=local_azimuths,
                    elevations=local_elevations,
                    azimuth_idx=azimuth_idx_right,
                    elevation_idx=elevation_idx_right,
                )
                return local_coords, ff
            else:
                elevation_idx_right = torch.bucketize(elevations, self.grid_ticks_elevation)
                mask_south_pole = elevation_idx_right == 0
                mask_north_pole = elevation_idx_right == num_elevations - 2
                mask_middle = ~mask_south_pole & ~mask_north_pole

                ff = torch.empty((*self.config.feature_dim, azimuths.size(0)), device=positions.device)
                local_positions = torch.empty((3, azimuths.size(0)), device=positions.device)
                local_azimuths = torch.empty(azimuths.size(0), device=positions.device)
                local_elevations = torch.empty(elevations.size(0), device=positions.device)
                omega_azimuth = self.grid_ticks_azimuth[1] - self.grid_ticks_azimuth[0]
                omega_elevation = self.grid_ticks_elevation[0] + pi_2

                # deal with the points in the middle
                # slerp between the two azimuths for the middle elevations
                azimuth_idx_left_middle = azimuth_idx_left[mask_middle]
                if len(azimuth_idx_left_middle) > 0:
                    azimuth_idx_right_middle = azimuth_idx_right[mask_middle]
                    elevation_idx_right_middle = elevation_idx_right[mask_middle]
                    elevation_idx_left_middle = elevation_idx_right_middle - 1
                    azimuth_idx_left_middle = azimuth_idx_left_middle.detach().cpu().numpy().tolist()
                    azimuth_idx_right_middle = azimuth_idx_right_middle.detach().cpu().numpy().tolist()
                    elevation_idx_left_middle = elevation_idx_left_middle.detach().cpu().numpy().tolist()
                    elevation_idx_right_middle = elevation_idx_right_middle.detach().cpu().numpy().tolist()
                    feature_slice = [
                        slice(None, None, None),
                        [
                            [azimuth_idx_left_middle, azimuth_idx_right_middle],
                            [azimuth_idx_left_middle, azimuth_idx_right_middle],
                        ],
                        [
                            [elevation_idx_left_middle, elevation_idx_left_middle],
                            [elevation_idx_right_middle, elevation_idx_right_middle],
                        ],
                    ]
                    local_coords, grid_values = self.position_interpolator(positions[:, mask_middle], feature_slice)
                    # grid_values has shape (f1, f2, ..., 2, 2, N)
                    thetas = azimuths[mask_middle] - self.grid_ticks_azimuth[azimuth_idx_left_middle]
                    f_bottom = slerp(  # (f1, f2, ..., N)
                        grid_values[..., 0, 0, :],
                        grid_values[..., 0, 1, :],
                        thetas,
                        omega_azimuth,
                        self.config.normalized,
                    )
                    f_top = slerp(  # (f1, f2, ..., N)
                        grid_values[..., 1, 0, :],
                        grid_values[..., 1, 1, :],
                        thetas,
                        omega_azimuth,
                        self.config.normalized,
                    )
                    local_positions[:, mask_middle] = local_coords
                    local_azimuths[mask_middle] = thetas / omega_azimuth
                    # slerp between the two elevations
                    thetas = elevations[mask_middle] - self.grid_ticks_elevation[elevation_idx_left_middle]
                    assert torch.all(thetas >= 0)
                    ff[..., mask_middle] = slerp(f_bottom, f_top, thetas, omega_elevation, self.config.normalized)
                    local_elevations[mask_middle] = thetas / omega_elevation

                # deal with the points at the south pole
                # slerp between the two azimuths for the south pole
                azimuth_idx_left_sp = azimuth_idx_left[mask_south_pole]
                if len(azimuth_idx_left_sp) > 0:
                    azimuth_idx_right_sp = azimuth_idx_right[mask_south_pole]
                    azimuth_idx_left_sp = azimuth_idx_left_sp.detach().cpu().numpy().tolist()
                    azimuth_idx_right_sp = azimuth_idx_right_sp.detach().cpu().numpy().tolist()
                    feature_slice = [slice(None, None, None), [azimuth_idx_left_sp, azimuth_idx_right_sp], 0]
                    positions_sp = positions[:, mask_south_pole]
                    local_coords, grid_values = self.position_interpolator(positions_sp, feature_slice)
                    # grid_values has shape (f1, f2, ..., 2, N)
                    thetas = azimuths[mask_south_pole] - self.grid_ticks_azimuth[azimuth_idx_left_sp]
                    f = slerp(  # (f1, f2, ..., N)
                        grid_values[..., 0, :],
                        grid_values[..., 1, :],
                        thetas,
                        omega_azimuth,
                        self.config.normalized,
                    )
                    local_positions[:, mask_south_pole] = local_coords
                    local_azimuths[mask_south_pole] = thetas / omega_azimuth
                    # slerp between the two elevations
                    local_coords, f_sp = self.pole_interpolator(positions_sp, [slice(None, None, None), 0])
                    thetas = elevations[mask_south_pole] + pi_2
                    assert torch.all(thetas >= 0)
                    ff[..., mask_south_pole] = slerp(f_sp, f, thetas, omega_elevation, self.config.normalized)
                    local_elevations[mask_south_pole] = thetas / omega_elevation

                # deal with the points at the north pole
                # slerp between the two azimuths for the north pole
                azimuth_idx_left_np = azimuth_idx_left[mask_north_pole]
                if len(azimuth_idx_left_np) > 0:
                    azimuth_idx_right_np = azimuth_idx_right[mask_north_pole]
                    azimuth_idx_left_np = azimuth_idx_left_np.detach().cpu().numpy().tolist()
                    azimuth_idx_right_np = azimuth_idx_right_np.detach().cpu().numpy().tolist()
                    feature_slice = [slice(None, None, None), [azimuth_idx_left_np, azimuth_idx_right_np], -1]
                    positions_np = positions[:, mask_north_pole]
                    local_coords, grid_values = self.position_interpolator(positions_np, feature_slice)
                    # grid_values has shape (f1, f2, ..., 2, N)
                    thetas = azimuths[mask_north_pole] - self.grid_ticks_azimuth[azimuth_idx_left_np]
                    f = slerp(  # (f1, f2, ..., N)
                        grid_values[..., 0, :],
                        grid_values[..., 1, :],
                        thetas,
                        omega_azimuth,
                        self.config.normalized,
                    )
                    local_positions[:, mask_north_pole] = local_coords
                    local_azimuths[mask_north_pole] = thetas / omega_azimuth
                    # slerp between the two elevations
                    local_coords, f_np = self.pole_interpolator(positions_np, [slice(None, None, None), 1])
                    thetas = pi_2 - elevations[mask_north_pole]
                    assert torch.all(thetas >= 0)
                    ff[..., mask_north_pole] = slerp(f_np, f, thetas, omega_elevation, self.config.normalized)
                    local_elevations[mask_north_pole] = thetas / omega_elevation

                local_coords = dict(
                    positions=local_positions,
                    azimuths=local_azimuths,
                    elevations=local_elevations,
                    azimuth_idx=azimuth_idx_right,
                    elevation_idx=elevation_idx_right,
                )
                return local_coords, ff
