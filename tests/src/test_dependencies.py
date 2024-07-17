import cv2


def test_dependencies():
    """Test dependencies.
    for check dependencie's library version.
    """
    version = cv2.__version__
    assert version == "4.10.0"
