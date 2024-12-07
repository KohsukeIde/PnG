from PIL import Image, ImageChops

def are_images_identical(image_path1, image_path2):
    """
    Compare two images pixel by pixel.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :return: True if images are identical, False otherwise.
    """
    try:
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
    except IOError as e:
        print(f"Error opening images: {e}")
        return False

    # Check if sizes are the same
    if img1.size != img2.size:
        print("Images have different sizes.")
        return False

    # Check if modes are the same (e.g., RGB, RGBA)
    if img1.mode != img2.mode:
        print("Images have different modes.")
        return False

    # Compute the difference
    diff = ImageChops.difference(img1, img2)

    if not diff.getbbox():
        # No differences
        return True
    else:
        # Differences found
        return False

# Example usage
image1 = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022_shifted.png'
image2 = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022_shifted2.png'

if are_images_identical(image1, image2):
    print("Images are identical.")
else:
    print("Images are different.")
