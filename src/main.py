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
    compute_homography,
    build_reference_data
)

folder_sign_types = {
    "0": "Alárendelt utak kereszteződése",
    "1": "100-as sebességkorlátozó tábla",
    "2": "80-as sebességkorlátozó tábla",
    "3": "Főútvonal",
    "4": "STOP tábla",
    "5": "Körforgalom"
}


def main():
    reference_dir = "../DATA"
    reference_data = build_reference_data(reference_dir)

    test_dir = "../TEST_DATA"
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not test_files:
        print("Nincs teszt kép a TEST_DATA mappában.")
        return

    random_file = random.choice(test_files)
    file_path = os.path.join(test_dir, random_file)
    image = load_image(file_path)
    if image is None:
        print(f"Nem sikerült betölteni a teszt képet: {file_path}")
        return

    # Nem vágjuk ki a táblát, az egész képet használjuk
    test_keypoints, test_descriptors = extract_sift_features(image)

    hsv_image = convert_to_hsv(image)
    masks = detect_color(hsv_image)
    dominant_color = classify_color(masks)

    MIN_MATCH_COUNT = 10
    best_match_type = None
    best_match_file = None
    best_inliers = 0
    best_good_matches = []

    for sign_type, ref_list in reference_data.items():
        for ref in ref_list:
            ref_keypoints = ref["keypoints"]
            ref_descriptors = ref["descriptors"]
            good_matches = match_features(test_descriptors, ref_descriptors, ratio_thresh=0.75)

            if len(good_matches) >= MIN_MATCH_COUNT:
                H, mask = compute_homography(test_keypoints, ref_keypoints, good_matches)
                if H is not None and mask is not None:
                    inlier_count = int(np.sum(mask))
                    if inlier_count > best_inliers:
                        best_inliers = inlier_count
                        best_match_type = sign_type
                        best_match_file = ref["filename"]
                        best_good_matches = good_matches

    if best_match_type:
        readable_sign_type = folder_sign_types.get(best_match_type, best_match_type)
        confidence = best_inliers / len(best_good_matches) if best_good_matches else 0.0
    else:
        readable_sign_type = "Nem azonosítható"
        confidence = 0.0

    print(f"Tesztkép: {random_file}")
    print(f"A domináns szín: {dominant_color}")
    print(f"A tábla típusa: {readable_sign_type}, \nKonfidencia: {confidence:.2f}")

    cv2.imshow("Tesztkép", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 27:
        print("Kiléptél az Esc gomb megnyomásával.")

if __name__ == "__main__":
    main()
