import os
import random
import cv2
import numpy as np
from utils import (
    load_image,
    convert_to_hsv,
    detect_color,
    classify_color,
    extract_sift_features,
    match_features,
    extract_orb_features,         
    match_orb_features,           
    compute_homography,
    build_reference_data,
    build_reference_data_orb      
)

folder_sign_types = {
    "alarendelt_utak": "Alárendelt utak kereszteződése",
    "100km": "100-as sebességkorlátozó tábla",
    "80km": "80-as sebességkorlátozó tábla",
    "fout": "Főútvonal",
    "stop": "STOP tábla",
    "korforgalom": "Körforgalom"
}

def main():
    reference_dir = "../DATA"
    reference_data_sift = build_reference_data(reference_dir)
    reference_data_orb = build_reference_data_orb(reference_dir)

    test_dir = "../TEST_DATA"
    all_files = os.listdir(test_dir)
    test_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not test_files:
        print("Nincs teszt kép a TEST_DATA mappában.")
        return

    random_file = random.choice(test_files)
    file_path = os.path.join(test_dir, random_file)
    image = load_image(file_path)
    if image is None:
        print(f"Nem sikerült betölteni a teszt képet: {file_path}")
        return

    hsv_image = convert_to_hsv(image)
    masks = detect_color(hsv_image)
    dominant_color = classify_color(masks)

    MIN_MATCH_COUNT = 10
    RATIO_THRESH = 0.85

    # ----------------------- SIFT Pipeline -----------------------
    test_keypoints_sift, test_descriptors_sift = extract_sift_features(image)
    best_match_type_sift = None
    best_inliers_sift = 0
    best_good_matches_sift = []

    for sign_type, ref_list in reference_data_sift.items():
        for ref in ref_list:
            ref_keypoints = ref["keypoints"]
            ref_descriptors = ref["descriptors"]
            good_matches = match_features(test_descriptors_sift, ref_descriptors, ratio_thresh=RATIO_THRESH)
            if len(good_matches) >= MIN_MATCH_COUNT:
                H, mask = compute_homography(test_keypoints_sift, ref_keypoints, good_matches)
                if H is not None and mask is not None:
                    inlier_count = int(np.sum(mask))
                    if inlier_count > best_inliers_sift:
                        best_inliers_sift = inlier_count
                        best_match_type_sift = sign_type
                        best_good_matches_sift = good_matches

    if best_match_type_sift:
        readable_sift = folder_sign_types.get(best_match_type_sift, best_match_type_sift)
        confidence_sift = best_inliers_sift / len(best_good_matches_sift) if best_good_matches_sift else 0.0
    else:
        readable_sift = "Nem azonosítható"
        confidence_sift = 0.0

    # ----------------------- ORB Pipeline -----------------------
    test_keypoints_orb, test_descriptors_orb = extract_orb_features(image)
    best_match_type_orb = None
    best_inliers_orb = 0
    best_good_matches_orb = []

    for sign_type, ref_list in reference_data_orb.items():
        for ref in ref_list:
            ref_keypoints = ref["keypoints"]
            ref_descriptors = ref["descriptors"]
            good_matches = match_orb_features(test_descriptors_orb, ref_descriptors, ratio_thresh=0.75)
            if len(good_matches) >= MIN_MATCH_COUNT:
                H, mask = compute_homography(test_keypoints_orb, ref_keypoints, good_matches)
                if H is not None and mask is not None:
                    inlier_count = int(np.sum(mask))
                    if inlier_count > best_inliers_orb:
                        best_inliers_orb = inlier_count
                        best_match_type_orb = sign_type
                        best_good_matches_orb = good_matches

    if best_match_type_orb:
        readable_orb = folder_sign_types.get(best_match_type_orb, best_match_type_orb)
        confidence_orb = best_inliers_orb / len(best_good_matches_orb) if best_good_matches_orb else 0.0
    else:
        readable_orb = "Nem azonosítható"
        confidence_orb = 0.0

    # ----------------------- Print Results -----------------------
    print("********** SIFT Pipeline Output **********")
    print(f"Tesztkép: {random_file}")
    print(f"A domináns szín: {dominant_color}")
    print(f"A tábla típusa: {readable_sift}, Konfidencia: {confidence_sift * 100:.2f}%")
    print("******************************************\n")

    print("********** ORB Pipeline Output **********")
    print(f"Tesztkép: {random_file}")
    print(f"A domináns szín: {dominant_color}")
    print(f"A tábla típusa: {readable_orb}, Konfidencia: {confidence_orb * 100:.2f}%")
    print("******************************************")

    cv2.imshow("Tesztkép", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
        print("Kiléptél az Esc gomb megnyomásával.")

if __name__ == "__main__":
    main()
