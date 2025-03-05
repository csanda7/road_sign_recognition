import cv2 as cv2
from utils import load_image, convert_to_grayscale, detect_edges, show_image

# Load image
image = load_image("data/37/037_1_0001.png")

# Convert to grayscale
gray_image = convert_to_grayscale(image)

# Detect edges
edges = detect_edges(gray_image)

# Show results
show_image("Edges", edges)
