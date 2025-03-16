import cv2 as cv2
import os
from utils import (
    load_image
)

#   Colors conversion to HSV instead of RGB:
#   the red color is represented between: 0-179
#   the blue color is represented between: 100-130
#   the yellow color is represented between: 20-30


#Root directory
root_dir = "DATA"
kor = 0
haromszog = 0
negyszog = 0
hatszog = 0
nem_ismert = 0

# Go through all the files in the directories
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith((".png", "jpg", "jpeg")):  # Deal with the .png .jpg .jpeg files only
            file_path = os.path.join(subdir, file)
            #print(f"Feldolgozás: {file_path}")

            # Load and using canny on the image
            image = load_image(file_path)
            edges = cv2.Canny(image, 100, 200)
            contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour:  # Ellenőrizzük, hogy van-e legalább egy kontúr
                largest_edges = max(contour, key=cv2.contourArea)  # Kiválasztjuk a legnagyobb kontúrt
                perimeter = cv2.arcLength(largest_edges, True) 
                #print(f"Perimeter: {perimeter}")
            
            approx = cv2.approxPolyDP(largest_edges, 0.025 * perimeter, True)
            sides = len(approx)

          

            if sides == 3:
                shape = "Háromszög"
                haromszog += 1
            elif sides == 4:
                shape = "Négyszög"
                negyszog += 1
            elif sides == 6:
                shape = "Hatszög"
                hatszog += 1
            elif sides >= 8:
                shape = "Kör"
                kor += 1
            else:
                shape = "Nem ismert"
                nem_ismert += 1

            #print(f"A tábla alakja: {shape}")
            #SHow the iamnge
            #cv2.imshow(f"Feldolgozva - {file}", edges)

            # Press Esc to exit
            # key = cv2.waitKey(0)  
            # cv2.destroyAllWindows()
            
            # if key == 27:  # ASCII code for Esc key
            #     print("Kiléptél a programból az Esc megnyomásával.")
            #     exit()
print(f"Körök száma: {kor}")
print(f"Háromszögek száma: {haromszog}")    
print(f"Négyszögek száma: {negyszog}")
print(f"Hatszögek száma: {hatszog}")
print(f"Nem ismert alakzatok száma: {nem_ismert}")