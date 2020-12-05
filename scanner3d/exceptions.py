"""
Defines custom exceptions for Scanner3D
"""


class NoDeviceDetectedException(Exception):
    """Raised when no device could be detected"""

    pass


class PointCloudSizeMismatch(Exception):
    """Raised when len(pcd1.points) != len(pcd2.points)"""

    pass
