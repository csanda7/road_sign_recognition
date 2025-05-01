#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import (
    load_image,
    extract_sift_features,
    extract_orb_features,
    match_features,
    match_orb_features,
    compute_homography,
    build_reference_data,
    build_reference_data_orb
)

MIN_MATCH_COUNT = 10
RATIO_THRESH = 0.85

"""
This script evaluates reference vs. test images using SIFT and ORB,
then plots counts and average confidences per class.
"""

def classify(img, refs, extract_fn, match_fn):
    kps, desc = extract_fn(img)
    best_label, best_inliers, best_matches = "Nem azonosítható", 0, 0
    for label, ref_list in refs.items():
        for ref in ref_list:
            matches = match_fn(desc, ref["descriptors"], RATIO_THRESH)
            if len(matches) < MIN_MATCH_COUNT:
                continue
            H, mask = compute_homography(kps, ref["keypoints"], matches)
            if mask is None:
                continue
            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers, best_matches, best_label = inliers, len(matches), label
    conf = best_inliers / best_matches if best_matches else 0
    return best_label, conf


def evaluate(ref_dir, test_dir, out_dir):
    refs = {
        "SIFT": build_reference_data(ref_dir),
        "ORB": build_reference_data_orb(ref_dir)
    }
    stats = {name: {} for name in refs}

    paths = [p for p in Path(test_dir).rglob("*")
             if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    for path in tqdm(paths, desc="Képek vizsgálata", unit="img"):
        img = load_image(str(path))
        if img is None:
            continue
        for name, extract_fn, match_fn in [
            ("SIFT", extract_sift_features, match_features),
            ("ORB", extract_orb_features, match_orb_features)
        ]:
            label, conf = classify(img, refs[name], extract_fn, match_fn)
            entry = stats[name].setdefault(label, {"count": 0, "sum": 0})
            entry["count"] += 1
            entry["sum"] += conf

    classes = sorted(set(stats["SIFT"]) | set(stats["ORB"]))
    x = np.arange(len(classes))
    width = 0.35
    counts = []
    avgs = []
    for name in ("SIFT", "ORB"):
        counts.append([stats[name].get(c, {"count": 0})["count"] for c in classes])
        avgs.append([
            (stats[name].get(c, {"sum": 0})["sum"] / stats[name].get(c, {"count": 0})["count"]
             if stats[name].get(c, {"count": 0})["count"] else 0)
            for c in classes
        ])

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Plot counts per sign
    plt.bar(x - width/2, counts[0], width, label="SIFT")
    plt.bar(x + width/2, counts[1], width, label="ORB")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.title("Felismert táblák száma")
    plt.ylabel("Darabszám")
    plt.xlabel("Tábla típus")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output / "counts_per_sign.png")
    plt.clf()

    # Plot average confidence per sign (excluding unknown)
    exclude = "Nem azonosítható"
    avg_classes = [c for c in classes if c != exclude]
    idx = [classes.index(c) for c in avg_classes]
    x2 = np.arange(len(avg_classes))
    avgs_filtered = []
    for series in avgs:
        avgs_filtered.append([series[i] for i in idx])

    plt.bar(x2 - width/2, avgs_filtered[0], width, label="SIFT")
    plt.bar(x2 + width/2, avgs_filtered[1], width, label="ORB")
    plt.xticks(x2, avg_classes, rotation=45, ha="right")
    plt.title("Átlagos konfidenciaszint tábla típusonként")
    plt.ylabel("Konfidenciaszint")
    plt.xlabel("Tábla típus")
    plt.axhline(0.5, color="red", linestyle="--", label="50% küszöb")
    plt.axhline(0.75, color="orange", linestyle="--", label="75% küszöb")
    plt.legend(loc="best")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output / "avg_conf_per_sign.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="../DATA")
    parser.add_argument("--test", default="../TEST_DATA")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    evaluate(args.ref, args.test, args.out)
    print("Vizsgálat befejeződött.")
    print("Eredmények mentve az", args.out, "mappába.")
