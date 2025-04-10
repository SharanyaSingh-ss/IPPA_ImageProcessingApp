import cv2
import numpy as np
from PIL import Image

# Function to apply grayscale filter
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply Gaussian blur
def apply_gaussian_blur(image, intensity=15):
    return cv2.GaussianBlur(image, (intensity, intensity), 0)

# Function to apply sepia filter
def apply_sepia(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter).clip(0, 255).astype(np.uint8)

    # Convert back to RGB for proper display in Streamlit
    return cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB)

# Function to apply sharpening filter
def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Function to apply cartoon effect
def apply_cartoon_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)