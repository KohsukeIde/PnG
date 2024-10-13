import torch

class CameraModel:
    def __init__(self, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        """
        Initialize the CameraModel.

        Args:
            intrinsics (torch.Tensor): Camera intrinsic parameters (3x3 matrix)
            extrinsics (torch.Tensor): Camera extrinsic parameters (4x4 matrix)

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if intrinsics.shape != (3, 3):
            raise ValueError("Intrinsics must be a 3x3 tensor")
        if extrinsics.shape != (4, 4):
            raise ValueError("Extrinsics must be a 4x4 tensor")

        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def project_3d_to_2d(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Project 3D points to 2D image plane.

        Args:
            points_3d (torch.Tensor): 3D points in world coordinates (Nx3)

        Returns:
            torch.Tensor: 2D points in image coordinates (Nx2)
        """
        # Ensure points_3d is homogeneous
        if points_3d.shape[1] == 3:
            points_3d = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)

        # Transform points to camera coordinates
        points_cam = torch.matmul(self.extrinsics, points_3d.t()).t()

        # Project to image plane
        points_2d = torch.matmul(self.intrinsics, points_cam[:, :3].t()).t()

        # Normalize
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]

        return points_2d

    def back_project_2d_to_3d(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Back-project 2D points to 3D space.

        Args:
            points_2d (torch.Tensor): 2D points in image coordinates (Nx2)
            depth (torch.Tensor): Depth values for each point (N)

        Returns:
            torch.Tensor: 3D points in world coordinates (Nx3)
        """
        # Create homogeneous coordinates
        points_2d_h = torch.cat([points_2d, torch.ones_like(points_2d[:, :1])], dim=1)

        # Invert intrinsics
        inv_intrinsics = torch.inverse(self.intrinsics)

        # Back-project to camera space
        points_cam = torch.matmul(inv_intrinsics, points_2d_h.t()).t()
        points_cam *= depth.unsqueeze(1)

        # Transform to world coordinates
        points_3d = torch.matmul(torch.inverse(self.extrinsics), 
                                 torch.cat([points_cam, torch.ones_like(points_cam[:, :1])], dim=1).t()).t()

        return points_3d[:, :3]

    def update_extrinsics(self, rotation: torch.Tensor, translation: torch.Tensor):
        """
        Update camera extrinsics.

        Args:
            rotation (torch.Tensor): 3x3 rotation matrix
            translation (torch.Tensor): 3x1 translation vector

        Raises:
            ValueError: If input tensors have incorrect shapes
        """
        if rotation.shape != (3, 3):
            raise ValueError("Rotation must be a 3x3 tensor")
        if translation.shape != (3, 1) and translation.shape != (3,):
            raise ValueError("Translation must be a 3x1 or 3-element tensor")

        translation = translation.view(3, 1)
        new_extrinsics = torch.eye(4, device=rotation.device)
        new_extrinsics[:3, :3] = rotation
        new_extrinsics[:3, 3] = translation.squeeze()

        self.extrinsics = new_extrinsics