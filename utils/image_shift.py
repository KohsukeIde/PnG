#!/usr/bin/env python3
# image_shift.py

import numpy as np
import cv2
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Shift an image by 1 pixel in the specified direction.')
    parser.add_argument('input_image', type=str, help='Path to the input image.')
    parser.add_argument('output_image', type=str, help='Path to save the shifted image.')
    parser.add_argument('--direction', type=str, default='right',
                        choices=['left', 'right', 'up', 'down'],
                        help='Direction to shift the image. Choices are left, right, up, down. Default is right.')
    return parser.parse_args()

def shift_image(image, direction):
    """
    Shift the image by 1 pixel in the specified direction.
    
    Args:
        image (numpy.ndarray): Input image.
        direction (str): Direction to shift ('left', 'right', 'up', 'down').
    
    Returns:
        shifted_image (numpy.ndarray): Shifted image.
    """
    # Define the translation matrix
    if direction == 'right':
        M = np.float32([[1, 0, 1],  # Shift right by 1
                        [0, 1, 0]])
    elif direction == 'left':
        M = np.float32([[1, 0, -1],  # Shift left by 1
                        [0, 1, 0]])
    elif direction == 'up':
        M = np.float32([[1, 0, 0],
                        [0, 1, -1]])  # Shift up by 1
    elif direction == 'down':
        M = np.float32([[1, 0, 0],
                        [0, 1, 1]])  # Shift down by 1
    else:
        raise ValueError("Invalid direction. Choose from 'left', 'right', 'up', 'down'.")

    height, width = image.shape[:2]
    
    # Apply the translation
    shifted_image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return shifted_image

def main():
    args = parse_arguments()
    
    # Load the input image
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Unable to load image at '{args.input_image}'. Please check the path.")
        sys.exit(1)
    
    # Shift the image
    shifted_image = shift_image(image, args.direction)
    
    # Save the shifted image
    success = cv2.imwrite(args.output_image, shifted_image)
    if not success:
        print(f"Error: Failed to save shifted image to '{args.output_image}'.")
        sys.exit(1)
    
    print(f"Successfully shifted the image '{args.input_image}' to the '{args.direction}' direction and saved as '{args.output_image}'.")

if __name__ == '__main__':
    main()
