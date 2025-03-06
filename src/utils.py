import cv2
import os


#Defining the functions

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    return cv2.imread(image_path)

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_threshold(image, threshold=127):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image, contours):
    return cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
