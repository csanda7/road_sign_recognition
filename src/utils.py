import cv2
import os
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nem sikerült betölteni a képet: {image_path}")
    return image

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def detect_color(hsv_image):

    masks = {}

    # Piros szín – két tartomány:
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    masks["Piros"] = cv2.bitwise_or(mask_red1, mask_red2)

    # Kék szín
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    masks["Kék"] = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Zöld szín
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    masks["Zöld"] = cv2.inRange(hsv_image, lower_green, upper_green)

    # Sárga szín
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    masks["Sárga"] = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    return masks

def classify_color(masks):
    max_color = None
    max_count = 0
    for color, mask in masks.items():
        count = cv2.countNonZero(mask)
        if count > max_count:
            max_count = count
            max_color = color
    return max_color if max_color is not None else "Ismeretlen"

def extract_sift_features(image):

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.7):
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def compute_homography(keypoints1, keypoints2, matches, ransac_threshold=8.0):
    if len(matches) < 4:
        print("Nincs elegendő egyezés.")
        return None, None

    source_points = []
    for match in matches:
        point = keypoints1[match.queryIdx].pt
        source_points.append(point)

    destination_points = []
    for match in matches:
        point = keypoints2[match.trainIdx].pt
        destination_points.append(point)

    source_points = np.array(source_points, dtype=np.float32)
    destination_points = np.array(destination_points, dtype=np.float32)

    source_points = source_points.reshape(-1, 1, 2)
    destination_points = destination_points.reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, ransac_threshold)

    return homography_matrix, mask

def build_reference_data(reference_dir):
    reference_data = {}
    for folder in os.listdir(reference_dir):
        folder_path = os.path.join(reference_dir, folder)
        if os.path.isdir(folder_path):
            sign_type = folder  
            reference_data[sign_type] = []
            for file in os.listdir(folder_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(folder_path, file)
                    image = load_image(image_path)
                    if image is not None:
                        kp, desc = extract_sift_features(image)
                        reference_data[sign_type].append({
                            "filename": file,
                            "keypoints": kp,
                            "descriptors": desc
                        })
    return reference_data


# ------------------------- ORB Functions -------------------------

def extract_orb_features(image):

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_orb_features(desc1, desc2, ratio_thresh=0.75):
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue
        m, n = match
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    return good_matches

def build_reference_data_orb(reference_dir):

    reference_data = {}
    for folder in os.listdir(reference_dir):
        folder_path = os.path.join(reference_dir, folder)
        if os.path.isdir(folder_path):
            sign_type = folder  
            reference_data[sign_type] = []
            for file in os.listdir(folder_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(folder_path, file)
                    image = load_image(image_path)
                    if image is not None:
                        kp, desc = extract_orb_features(image)
                        reference_data[sign_type].append({
                            "filename": file,
                            "keypoints": kp,
                            "descriptors": desc
                        })
    return reference_data
