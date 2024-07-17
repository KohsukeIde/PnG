import cv2


def test_dependencies():
    version = cv2.__version__
    assert version == "4.10.0"
    
