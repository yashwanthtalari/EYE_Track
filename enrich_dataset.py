"""
Enrich an image dataset with geometric + eye-strip features -- no recollection.

Reads dataset/labels.csv, re-runs MediaPipe Face Mesh on each already-saved face
crop, and extracts the feature blocks defined in gaze_features.py. The result is
cached to a single .npz so training can iterate instantly without touching the
webcam or re-detecting faces every run.

Usage:
    python enrich_dataset.py
    python enrich_dataset.py --data dataset --out dataset/features_cache.npz
"""
import cv2
import numpy as np
import pandas as pd
import argparse
import os

from detector import FaceMeshDetector
from gaze_features import extract_from_crop, DEFAULT_EYE_SIZE, GEOM_DIM, eye_dim


def parse_args():
    p = argparse.ArgumentParser(description="Cache geometric + eye-strip features for the gaze dataset.")
    p.add_argument("--data", default="dataset", help="Dataset folder from collect_data.py (default: dataset)")
    p.add_argument("--out", default="dataset/features_cache.npz", help="Output .npz cache path")
    p.add_argument("--eye-w", type=int, default=DEFAULT_EYE_SIZE[0], help="Eye patch width (default: 32)")
    p.add_argument("--eye-h", type=int, default=DEFAULT_EYE_SIZE[1], help="Eye patch height (default: 18)")
    return p.parse_args()


def main():
    args = parse_args()
    eye_size = (args.eye_w, args.eye_h)

    labels_path = os.path.join(args.data, "labels.csv")
    images_dir = os.path.join(args.data, "images")
    if not os.path.exists(labels_path):
        print(f"[Enrich] labels.csv not found in {args.data}. Run collect_data.py first.")
        return

    df = pd.read_csv(labels_path)
    detector = FaceMeshDetector()

    geoms, eyes, ys, sessions, points, names = [], [], [], [], [], []
    missing = no_face = 0
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False), 1):
        img_path = os.path.join(images_dir, str(row.image))
        bgr = cv2.imread(img_path)
        if bgr is None:
            missing += 1
            continue

        geom, eye = extract_from_crop(bgr, detector, eye_size)
        if geom is None or eye is None:
            no_face += 1
            continue

        geoms.append(geom)
        eyes.append(eye)
        ys.append([float(row.target_norm_x), float(row.target_norm_y)])
        sessions.append(str(row.session_id))
        points.append(int(row.point_index))
        names.append(str(row.image))

        if i % 200 == 0 or i == total:
            print(f"[Enrich] {i}/{total} processed  (kept={len(geoms)}, missing={missing}, no_face={no_face})")

    if not geoms:
        print("[Enrich] No features extracted -- nothing to save.")
        return

    screen_w = int(df["screen_w"].median())
    screen_h = int(df["screen_h"].median())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        geom=np.array(geoms, dtype=np.float32),
        eyes=np.array(eyes, dtype=np.float32),
        y=np.array(ys, dtype=np.float32),
        sessions=np.array(sessions),
        points=np.array(points, dtype=np.int32),
        names=np.array(names),
        eye_size=np.array(eye_size, dtype=np.int32),
        screen=np.array([screen_w, screen_h], dtype=np.int32),
    )
    print(f"\n[Enrich] Saved {len(geoms)} samples to {args.out}")
    print(f"[Enrich]   geom dim={GEOM_DIM}  eye dim={eye_dim(eye_size)}  screen={screen_w}x{screen_h}")
    print(f"[Enrich]   dropped: {missing} unreadable + {no_face} no-face/failed")


if __name__ == "__main__":
    main()
