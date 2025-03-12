import cv2 as cv2
import os
from utils import (
    load_image, convert_to_grayscale, detect_edges, 
    show_image, apply_gaussian_blur, find_contours, draw_contours
)

#Root directory
root_dir = "data"

# Go through all the files in the directories
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith((".png")):  # Deal with the .png files only
            file_path = os.path.join(subdir, file)
            print(f"Processing: {file_path}")

            # Load and edit the image
            image = load_image(file_path)
            gray_image = convert_to_grayscale(image)
            blurred = apply_gaussian_blur(gray_image)
            edges = detect_edges(blurred)
            found_contours = find_contours(edges)
            contour = draw_contours(image, found_contours)

            #SHow the iamnge
            cv2.imshow(f"Processed - {file}", contour)

            # Press Esc to exit
            key = cv2.waitKey(0)  
            cv2.destroyAllWindows()
            
            if key == 27:  # ASCII code for Esc key
                print("Esc key pressed. Exiting program.")
                exit()
