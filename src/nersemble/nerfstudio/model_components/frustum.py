import numpy as np
from typing import Tuple, List
import torch


class TorchHalfSpace3D:

    def __init__(self, normal: torch.tensor, offset: torch.tensor):
        self.normal = normal / normal.norm(p=2)
        self.offset = offset

    def signedDistance(self, point: torch.tensor):
        return self.normal.dot(point - self.offset)


class HalfSpace3D:

    def __init__(self, normal: np.ndarray, offset: np.ndarray):
        self.normal = normal / np.linalg.norm(normal)
        self.offset = offset

    def signedDistance(self, point: np.ndarray):
        return self.normal.dot(point - self.offset)


class TorchHalfSpaceCollection:
    def __init__(self, half_spaces: List[HalfSpace3D]):
        self.normals = torch.stack([hs.normal for hs in half_spaces])
        self.offsets = torch.stack([hs.offset for hs in half_spaces])
        self.offsets = self.offsets.cuda()
        self.normals = self.normals.cuda()

    def contains(self, point: np.ndarray) -> bool:
        point_reshaped = point.reshape((1, 1, 3))
        offsets_reshaped = self.offsets.reshape((1, 4, 3))
        diff = point_reshaped - offsets_reshaped
        diff_reshaped = diff.reshape(4, 1, 3)
        normals_reshaped = self.normals.unsqueeze(2)  # [4, 3, 1]
        signed_distances = diff_reshaped @ normals_reshaped  # [4, 1, 1]
        # signed_distances = self.normals @ (point - self.offsets).T
        return (signed_distances >= 0).all().item()

    def contains_points(self, points: torch.tensor) -> torch.tensor:
        B = points.shape[0]
        points_reshaped = points.unsqueeze(1)  # [B, 1, 3]
        offsets_reshaped = self.offsets.unsqueeze(0)  # [1, 4, 3]
        diff = (points_reshaped - offsets_reshaped)  # [B, 4, 3]
        diff_reshaped = diff.reshape(-1, 1, 3)  # [B*4, 1, 3]

        normals_reshaped = self.normals.repeat((B, 1)).unsqueeze(2)  # [B*4, 3, 1]
        signed_distances = (diff_reshaped @ normals_reshaped).reshape(B, 4)
        visibility_mask = (signed_distances >= 0).all(dim=1)
        return visibility_mask


class HalfSpaceCollection:

    def __init__(self, half_spaces: List[HalfSpace3D]):
        self.normals = np.stack([hs.normal for hs in half_spaces])
        self.offsets = np.stack([hs.offset for hs in half_spaces])

    def contains(self, point: np.ndarray) -> bool:
        point_reshaped = point.reshape((1, 1, 3))
        offsets_reshaped = self.offsets.reshape((1, 4, 3))
        diff = point_reshaped - offsets_reshaped
        diff_reshaped = diff.reshape(4, 1, 3)
        normals_reshaped = np.expand_dims(self.normals, 2)  # [4, 3, 1]
        signed_distances = diff_reshaped @ normals_reshaped  # [4, 1, 1]
        # signed_distances = self.normals @ (point - self.offsets).T
        return np.all(signed_distances >= 0).item()

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        B = points.shape[0]
        points_reshaped = np.expand_dims(points, 1)  # [B, 1, 3]
        offsets_reshaped = np.expand_dims(self.offsets, 0)  # [1, 4, 3]
        diff = (points_reshaped - offsets_reshaped)  # [B, 4, 3]
        diff_reshaped = diff.reshape(-1, 1, 3)  # [B*4, 1, 3]

        # Have to use np.tile() instead of repeat() as repeat() does not "concat" multiple instances of the matrix, but duplicates rows
        # repeat(2):
        #             0 1
        #   0 1   ->  0 1
        #   2 3       2 3
        #             2 3
        #
        # tile():
        #             0 1
        #   0 1   ->  2 3
        #   2 3       0 1
        #             2 3
        # tile() has the disadvantage that it can only tile the last dimension. Hence, we have to first transpose a few times to have the correct dimension as last dim
        normals_reshaped = np.expand_dims(np.tile(self.normals.T, B).T, 2)  # [B*4, 3, 1]
        signed_distances = (diff_reshaped @ normals_reshaped).reshape(B, 4)
        visibility_mask = np.all(signed_distances >= 0, axis=1)
        return visibility_mask


class TorchFrustum:

    def __init__(self, cam_to_world: torch.tensor, intrinsics: torch.tensor, image_dimensions: Tuple[int, int]):
        depth = 1
        img_w, img_h = image_dimensions

        # Assume that x -> right, y -> down, z-> forward
        p_top_left = torch.tensor([0, 0, depth, 1], dtype=cam_to_world.dtype)
        p_top_right = torch.tensor([img_w * depth, 0, depth, 1], dtype=cam_to_world.dtype)
        p_bottom_left = torch.tensor([0, img_h * depth, depth, 1], dtype=cam_to_world.dtype)
        p_bottom_right = torch.tensor([img_w * depth, img_h * depth, depth, 1], dtype=cam_to_world.dtype)
        p_center = cam_to_world[:3, 3]

        points = torch.stack([p_top_left, p_top_right, p_bottom_left, p_bottom_right])
        intrinsics_homogenized = torch.eye(4, dtype=cam_to_world.dtype)
        intrinsics_homogenized[:3, :3] = torch.linalg.inv(intrinsics)
        points_world = (cam_to_world @ intrinsics_homogenized @ points.T).T
        p_top_left, p_top_right, p_bottom_left, p_bottom_right = points_world[:4, :3]

        # vectors marking the edges of the view frustum
        v_top_left = p_top_left - p_center
        v_top_right = p_top_right - p_center
        v_bottom_left = p_bottom_left - p_center
        v_bottom_right = p_bottom_right - p_center

        # Cross-product inherits handedness of coordinate system (right-handed in our case)
        # Hence, direction of normal can be inferred via the right-hand rule.
        # We want the normals to point inside the frustum
        normal_top = torch.cross(v_top_left, v_top_right)
        normal_right = torch.cross(v_top_right, v_bottom_right)
        normal_bottom = torch.cross(v_bottom_right, v_bottom_left)
        normal_left = torch.cross(v_bottom_left, v_top_left)

        self.half_spaces = [
            TorchHalfSpace3D(normal_top, p_center),
            TorchHalfSpace3D(normal_right, p_center),
            TorchHalfSpace3D(normal_bottom, p_center),
            TorchHalfSpace3D(normal_left, p_center),
        ]

        self._half_space_collection = TorchHalfSpaceCollection(self.half_spaces)

    def contains(self, point: torch.tensor) -> bool:
        return self._half_space_collection.contains(point)

    def contains_points(self, points: torch.tensor) -> torch.Tensor:
        return self._half_space_collection.contains_points(points)


class Frustum:

    def __init__(self, cam_to_world: np.ndarray, intrinsics: np.ndarray, image_dimensions: Tuple[int, int]):
        depth = 1
        img_w, img_h = image_dimensions

        # Assume that x -> right, y -> down, z-> forward
        p_top_left = np.array([0, 0, depth, 1])
        p_top_right = np.array([img_w * depth, 0, depth, 1])
        p_bottom_left = np.array([0, img_h * depth, depth, 1])
        p_bottom_right = np.array([img_w * depth, img_h * depth, depth, 1])
        p_center = cam_to_world[:3, 3]

        points = np.stack([p_top_left, p_top_right, p_bottom_left, p_bottom_right])
        intrinsics_homogenized = np.eye(4)
        intrinsics_homogenized[:3, :3] = np.linalg.inv(intrinsics)
        points_world = (cam_to_world @ intrinsics_homogenized @ points.T).T
        p_top_left, p_top_right, p_bottom_left, p_bottom_right = points_world[:4, :3]

        # vectors marking the edges of the view frustum
        v_top_left = p_top_left - p_center
        v_top_right = p_top_right - p_center
        v_bottom_left = p_bottom_left - p_center
        v_bottom_right = p_bottom_right - p_center

        # Cross-product inherits handedness of coordinate system (right-handed in our case)
        # Hence, direction of normal can be inferred via the right-hand rule.
        # We want the normals to point inside the frustum
        normal_top = np.cross(v_top_left, v_top_right)
        normal_right = np.cross(v_top_right, v_bottom_right)
        normal_bottom = np.cross(v_bottom_right, v_bottom_left)
        normal_left = np.cross(v_bottom_left, v_top_left)

        self.half_spaces = [
            HalfSpace3D(normal_top, p_center),
            HalfSpace3D(normal_right, p_center),
            HalfSpace3D(normal_bottom, p_center),
            HalfSpace3D(normal_left, p_center),
        ]

        self._half_space_collection = HalfSpaceCollection(self.half_spaces)

    def contains(self, point: np.ndarray) -> bool:
        return self._half_space_collection.contains(point)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        return self._half_space_collection.contains_points(points)
