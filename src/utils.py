import cv2
import os

def load_image(image_path):
    """Loads an image from a file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    return cv2.imread(image_path)

def save_image(image, save_path):
    """Saves an image to a file."""
    cv2.imwrite(save_path, image)

def resize_image(image, width, height):
    """Resizes an image to the specified dimensions."""
    return cv2.resize(image, (width, height))

def convert_to_grayscale(image):
    """Converts an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies Gaussian blur to an image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    """Applies Canny edge detection."""
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_threshold(image, threshold=127):
    """Applies binary thresholding."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary
def find_contours(image):
    """Finds contours in a binary image."""
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    """Draws contours on an image."""
    return cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
def show_image(title, image):
    """Displays an image until a key is pressed."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
