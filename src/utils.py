import cv2
import os
import numpy as np



def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    return cv2.imread(image_path)


def show_image(title, image):
    cv2.imshow(title, image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


#   Colors conversion to HSV instead of RGB:
#   the piros color is represented between: 0-179
#   the blue color is represented between: 100-130
#   the yellow color is represented between: 20-30

color_ranges = {
                "piros": [(0, 70, 50), (10, 255, 255)],     # piros (lower range)
                "piros2": [(170, 70, 50), (180, 255, 255)], # piros (upper range)
                "kék": [(100, 150, 50), (130, 255, 255)],
                "sárga": [(20, 100, 100), (30, 255, 255)]
            }


def detect_color(hsv_image):
    masks = {}
    
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        masks[color] = mask
    
    # Combine piros masks (since it has two ranges)
    masks["piros"] = cv2.bitwise_or(masks["piros"], masks["piros2"])
    del masks["piros2"]  # Remove pirosundant entry
    
    return masks

def classify_color(masks):
    max_pixels = 0
    dominant_color = "ismeretlen"
    
    for color, mask in masks.items():
        pixel_count = cv2.countNonZero(mask)  # Count white pixels in mask
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color = color
    
    return dominant_color



# def save_image(image, save_path):
#     cv2.imwrite(save_path, image)

# def resize_image(image, width, height):
#     return cv2.resize(image, (width, height))

# def convert_to_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# def apply_gaussian_blur(image, kernel_size=(5, 5)):
#     return cv2.GaussianBlur(image, kernel_size, 0)

# def detect_edges(image, low_threshold=50, high_threshold=150):
#     return cv2.Canny(image, low_threshold, high_threshold)

# def apply_threshold(image, threshold=127):
#     _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
#     return binary

# def find_edgess(image):
#     edgess, _ = cv2.findedgess(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return edgess

# def draw_edgess(image, edgess):
#     return cv2.drawedgess(image.copy(), edgess, -1, (0, 255, 0), 1)

    
