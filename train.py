"""
Offline trainer / evaluator for the VisionPoint gaze model.

Trains the from-scratch NumPy MLP (neural_regressor.py) on the latest
calibration session and reports held-out accuracy in pixels. Optionally
compares against the Ridge polynomial baseline so you can see the trade-off
without launching the webcam.

Usage:
    python train.py                 # train MLP on latest session, save model
    python train.py --compare       # also report the Ridge baseline
    python train.py --hidden 128 64 # custom hidden-layer sizes
"""
import argparse
import numpy as np
import pandas as pd

from database import CalibrationDatabase
from neural_regressor import GazeNeuralRegressor, MLP, StandardScaler, FEATURE_COLS


def load_session():
    db = CalibrationDatabase()
    records, columns = db.load_latest_session(include_test_pass=False)
    if not records:
        print("[Train] No calibration data found. Run a calibration first.")
        return None
    df = pd.DataFrame(records, columns=columns)
    print(f"[Train] Loaded {len(df)} samples from latest session.")
    return df


def pixel_errors(y_true, y_pred):
    d = np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))
    return d.mean(), np.median(d), d.max()


def evaluate_mlp(df, hidden, test_frac=0.2, seed=0):
    """Hold out a test split, train the MLP, report honest pixel error."""
    X = df[FEATURE_COLS].values.astype(np.float64)
    Y = df[["screen_x", "screen_y"]].values.astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(4, int(test_frac * len(X)))
    te, tr = idx[:n_test], idx[n_test:]

    xs = StandardScaler().fit(X[tr])
    ys = StandardScaler().fit(Y[tr])
    # Carve a small validation slice out of the training set for early stopping.
    n_val = max(4, int(0.15 * len(tr)))
    val, core = tr[:n_val], tr[n_val:]

    net = MLP([X.shape[1], *hidden, 2], l2=1e-4, seed=42)
    net.fit(
        xs.transform(X[core]), ys.transform(Y[core]),
        X_val=xs.transform(X[val]), y_val=ys.transform(Y[val]),
        epochs=3000, batch_size=32, lr=1e-3, patience=200, verbose=False,
    )
    pred = ys.inverse_transform(net.predict(xs.transform(X[te])))
    mean, med, mx = pixel_errors(Y[te], pred)
    print(f"[MLP]   held-out pixel error  mean={mean:6.1f}  median={med:6.1f}  max={mx:6.1f}")
    return mean


def evaluate_ridge(df, test_frac=0.2, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    X = df[FEATURE_COLS].values.astype(np.float64)
    Y = df[["screen_x", "screen_y"]].values.astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(4, int(test_frac * len(X)))
    te, tr = idx[:n_test], idx[n_test:]

    mx = make_pipeline(PolynomialFeatures(2), Ridge(alpha=5.0)).fit(X[tr], Y[tr, 0])
    my = make_pipeline(PolynomialFeatures(2), Ridge(alpha=5.0)).fit(X[tr], Y[tr, 1])
    pred = np.column_stack([mx.predict(X[te]), my.predict(X[te])])
    mean, med, mxe = pixel_errors(Y[te], pred)
    print(f"[Ridge] held-out pixel error  mean={mean:6.1f}  median={med:6.1f}  max={mxe:6.1f}")
    return mean


def main():
    ap = argparse.ArgumentParser(description="Train the VisionPoint gaze MLP.")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32],
                    help="Hidden layer sizes (default: 64 32)")
    ap.add_argument("--compare", action="store_true",
                    help="Also evaluate the Ridge polynomial baseline")
    ap.add_argument("--no-save", action="store_true",
                    help="Only evaluate; do not (re)train and save the deployed model")
    args = ap.parse_args()

    df = load_session()
    if df is None:
        return

    print("\n--- Held-out evaluation (20% test split) ---")
    evaluate_mlp(df, tuple(args.hidden))
    if args.compare:
        evaluate_ridge(df)

    if not args.no_save:
        print("\n--- Fitting deployable MLP on the full session ---")
        reg = GazeNeuralRegressor(hidden_layers=tuple(args.hidden))
        reg.train_from_db(verbose=False)


if __name__ == "__main__":
    main()
