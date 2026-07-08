"""
Train an image-based gaze regressor from data collected by collect_data.py.

Pipeline
--------
1. Read dataset/labels.csv, load each face-crop JPG.
2. Turn each image into a compact feature vector: convert to grayscale,
   resize to a small square (default 32x32), equalize, and flatten to [0,1].
   (No deep-learning framework is required -- this keeps the project's
   lightweight, from-scratch spirit and runs on CPU in seconds.)
3. Fit an sklearn MLPRegressor mapping image features -> normalized (x, y).
   Targets are normalized [0,1] so the model is screen-resolution independent.
4. Report accuracy on a held-out split, in both normalized and pixel units,
   and pickle the model + preprocessing config to models/gaze_image_model.pkl.

Usage:
    python train_image_model.py
    python train_image_model.py --data dataset --img-size 32 --model ridge
    python train_image_model.py --model mlp --hidden 256 128
"""
import cv2
import numpy as np
import pandas as pd
import argparse
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


def parse_args():
    p = argparse.ArgumentParser(description="Train an image-based gaze model from collected face crops.")
    p.add_argument("--data", default="dataset", help="Dataset folder produced by collect_data.py (default: dataset)")
    p.add_argument("--out", default="models/gaze_image_model.pkl", help="Output model path")
    p.add_argument("--img-size", type=int, default=32, help="Downscaled square size fed to the model (default: 32)")
    p.add_argument("--model", choices=["mlp", "ridge"], default="mlp", help="Regressor type (default: mlp)")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128],
                   help="MLP hidden layer sizes (default: 256 128)")
    p.add_argument("--test-frac", type=float, default=0.2, help="Held-out fraction for evaluation (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def image_to_features(bgr, img_size):
    """Grayscale -> resize -> histogram-equalize -> flatten to [0,1] floats."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    small = cv2.equalizeHist(small)          # normalize lighting across sessions
    return small.astype(np.float32).flatten() / 255.0


def load_dataset(data_dir, img_size):
    """Load features X and normalized targets Y from labels.csv + images/."""
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found in {data_dir}. Run collect_data.py first.")

    df = pd.read_csv(labels_path)
    X, Y, kept, missing = [], [], 0, 0
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, str(row["image"]))
        bgr = cv2.imread(img_path)
        if bgr is None:
            missing += 1
            continue
        X.append(image_to_features(bgr, img_size))
        Y.append([float(row["target_norm_x"]), float(row["target_norm_y"])])
        kept += 1

    if kept == 0:
        raise RuntimeError("No images could be loaded. Check the images/ folder.")
    if missing:
        print(f"[Train] Warning: {missing} images listed in labels.csv were missing on disk.")

    # Median screen size across rows -> used to translate error into pixels.
    screen_w = int(df["screen_w"].median())
    screen_h = int(df["screen_h"].median())
    print(f"[Train] Loaded {kept} images | feature dim = {len(X[0])} | screen ~ {screen_w}x{screen_h}")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), screen_w, screen_h


def build_model(args):
    if args.model == "ridge":
        # Ridge handles high-dim pixel inputs well and never overfits hard; a
        # solid, fast baseline. Wrapped for 2 independent outputs (x, y).
        return MultiOutputRegressor(Ridge(alpha=1.0))
    return MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden),
        activation="relu",
        solver="adam",
        alpha=1e-3,                 # L2 regularization
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=1500,
        early_stopping=True,        # hold out an internal val split
        n_iter_no_change=30,
        random_state=args.seed,
    )


def report(name, y_true, y_pred, screen_w, screen_h):
    """Print normalized MAE and Euclidean pixel error for a split."""
    mae = np.mean(np.abs(y_true - y_pred), axis=0)  # per-axis, normalized
    # Convert normalized error to pixels using screen dimensions.
    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    px_err = np.sqrt(dx ** 2 + dy ** 2)

    def r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"[{name}] norm MAE x={mae[0]:.4f} y={mae[1]:.4f} | "
          f"R^2 x={r2(y_true[:,0], y_pred[:,0]):.3f} y={r2(y_true[:,1], y_pred[:,1]):.3f} | "
          f"pixel err mean={px_err.mean():.1f} median={np.median(px_err):.1f}")


def main():
    args = parse_args()
    X, Y, screen_w, screen_h = load_dataset(args.data, args.img_size)

    if len(X) < 10:
        print(f"[Train] Only {len(X)} samples -- collect more data before training.")
        return

    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=args.test_frac, random_state=args.seed
    )
    print(f"[Train] Train={len(X_tr)}  Test={len(X_te)}  Model={args.model}")

    model = build_model(args)
    model.fit(X_tr, Y_tr)

    report("Train", Y_tr, model.predict(X_tr), screen_w, screen_h)
    report("Test ", Y_te, model.predict(X_te), screen_w, screen_h)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "model": model,
        "img_size": args.img_size,
        "model_type": args.model,
        "screen_w": screen_w,
        "screen_h": screen_h,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"[Train] Saved model to {args.out}")
    print("[Train] Predicts NORMALIZED (x, y) in [0,1]; multiply by screen size for pixels.")


if __name__ == "__main__":
    main()
