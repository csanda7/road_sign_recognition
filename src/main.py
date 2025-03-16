import cv2 as cv2
import os
from utils import (
    load_image,convert_to_hsv, classify_color, detect_color
)

#Root directory
root_dir = "DATA"
kor = 0
haromszog = 0
negyszog = 0
hatszog = 0
nem_ismert = 0
image_id =1

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
            
            approx = cv2.approxPolyDP(largest_edges, 0.02 * perimeter, True) # 2%-kal közelítjük a kontúrt
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

            print(f"#{image_id}")
            image_id += 1
            print(f"A tábla alakja: {shape}")


            hsv_image = convert_to_hsv(image)
            masks = detect_color(hsv_image)
            dominant_color = classify_color(masks)
            print(f"A domináns szín: {dominant_color}")
            #for color, mask in masks.items():
                #cv2.imshow(f"Maszk: {color}", hsv_image)


            #SHow the iamnge
            cv2.imshow(f"Feldolgozva - {file}", image)
            print("*************************************************")

            # Press Esc to exit
            key = cv2.waitKey(0)  
            cv2.destroyAllWindows()
            
            if key == 27:  # ASCII code for Esc key
                print("Kiléptél a programból az Esc megnyomásával.")
                exit()


print(f"Körök száma: {kor}") #Összesen 464 kör van a képeken
print(f"Háromszögek száma: {haromszog}")   #0 háromszög van a képeken 
print(f"Négyszögek száma: {negyszog}") #29 négyszög van a képeken
print(f"Hatszögek száma: {hatszog}") # 19 hatszög van a képeken
print(f"Nem ismert alakzatok száma: {nem_ismert}") # 0 nem ismert alakzat van a képeken