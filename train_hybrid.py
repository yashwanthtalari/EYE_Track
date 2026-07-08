"""
Train an image-based gaze regressor from the enriched feature cache, with an
HONEST, group-aware evaluation.

Why group-aware? The collector saves ~30 near-identical frames per screen point.
A plain random train/test split therefore leaves a near-duplicate of every test
frame in the training set, so the reported error is wildly optimistic. Instead we
evaluate with:

  * leave-point-out   (GroupKFold on point_index): every held-out point's frames
    are fully unseen -> measures spatial interpolation across the screen.
  * leave-session-out (if >1 session): train on some sessions, test on a whole
    held-out session -> measures cross-session generalization (the real world).

Feature sets (--feats):
  geom    -> 12 geometric features (iris ratios, EAR, head pose)   [head-robust]
  eyes    -> flattened left+right eye-strip pixels
  hybrid  -> both, concatenated                                    [default]

Usage:
    python train_hybrid.py                       # hybrid, honest eval, save model
    python train_hybrid.py --feats geom
    python train_hybrid.py --compare             # report all 3 feature sets
    python train_hybrid.py --model ridge
"""
import numpy as np
import argparse
import os
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


def parse_args():
    p = argparse.ArgumentParser(description="Honest group-aware training for the gaze model.")
    p.add_argument("--cache", default="dataset/features_cache.npz", help="Feature cache from enrich_dataset.py")
    p.add_argument("--out", default="models/gaze_hybrid_model.pkl", help="Output model path")
    p.add_argument("--feats", choices=["geom", "eyes", "hybrid"], default="hybrid", help="Feature set (default: hybrid)")
    p.add_argument("--model", choices=["mlp", "ridge"], default="mlp", help="Regressor (default: mlp)")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128], help="MLP hidden sizes (default: 256 128)")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold folds for leave-point-out (default: 5)")
    p.add_argument("--compare", action="store_true", help="Evaluate geom/eyes/hybrid side by side, then save the chosen --feats model")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def build_X(cache, feats):
    if feats == "geom":
        return cache["geom"]
    if feats == "eyes":
        return cache["eyes"]
    return np.concatenate([cache["geom"], cache["eyes"]], axis=1)


def make_model(args):
    if args.model == "ridge":
        return MultiOutputRegressor(Ridge(alpha=1.0))
    return MLPRegressor(
        hidden_layer_sizes=tuple(args.hidden),
        activation="relu", solver="adam", alpha=1e-3,
        batch_size=32, learning_rate_init=1e-3,
        max_iter=1500, early_stopping=True, n_iter_no_change=30,
        random_state=args.seed,
    )


def pixel_errors(y_true, y_pred, screen_w, screen_h):
    dx = (y_true[:, 0] - y_pred[:, 0]) * screen_w
    dy = (y_true[:, 1] - y_pred[:, 1]) * screen_h
    return np.sqrt(dx ** 2 + dy ** 2)


def grouped_cv(X, Y, groups, args, screen_w, screen_h, label):
    """Group K-fold CV: fit a fresh scaler+model per fold, collect out-of-fold error."""
    n_groups = len(np.unique(groups))
    folds = min(args.folds, n_groups)
    if folds < 2:
        print(f"[{label}] Not enough groups ({n_groups}) for CV -- skipped.")
        return None
    gkf = GroupKFold(n_splits=folds)
    errs = []
    for tr, te in gkf.split(X, Y, groups):
        scaler = StandardScaler().fit(X[tr])
        model = make_model(args)
        model.fit(scaler.transform(X[tr]), Y[tr])
        pred = model.predict(scaler.transform(X[te]))
        errs.append(pixel_errors(Y[te], pred, screen_w, screen_h))
    errs = np.concatenate(errs)
    print(f"[{label}] leave-point-out ({folds}-fold): "
          f"pixel err mean={errs.mean():.1f}  median={np.median(errs):.1f}  p90={np.percentile(errs,90):.1f}")
    return errs.mean()


def leave_session_out(X, Y, sessions, args, screen_w, screen_h, label):
    uniq = np.unique(sessions)
    if len(uniq) < 2:
        return None
    errs = []
    for s in uniq:
        te = sessions == s
        tr = ~te
        scaler = StandardScaler().fit(X[tr])
        model = make_model(args)
        model.fit(scaler.transform(X[tr]), Y[tr])
        pred = model.predict(scaler.transform(X[te]))
        e = pixel_errors(Y[te], pred, screen_w, screen_h)
        errs.append(e)
        print(f"[{label}]   holdout {s}: mean={e.mean():.1f}  median={np.median(e):.1f}")
    allerr = np.concatenate(errs)
    print(f"[{label}] leave-session-out: pixel err mean={allerr.mean():.1f}  median={np.median(allerr):.1f}")
    return allerr.mean()


def main():
    args = parse_args()
    if not os.path.exists(args.cache):
        print(f"[Train] Cache not found: {args.cache}. Run enrich_dataset.py first.")
        return

    cache = np.load(args.cache, allow_pickle=True)
    Y = cache["y"]
    groups = cache["points"]
    sessions = cache["sessions"]
    screen_w, screen_h = [int(v) for v in cache["screen"]]
    eye_size = tuple(int(v) for v in cache["eye_size"])
    print(f"[Train] {len(Y)} samples | screen {screen_w}x{screen_h} | "
          f"{len(np.unique(groups))} points | {len(np.unique(sessions))} sessions")

    feat_sets = ["geom", "eyes", "hybrid"] if args.compare else [args.feats]
    for feats in feat_sets:
        X = build_X(cache, feats)
        print(f"\n=== feats={feats}  (dim={X.shape[1]})  model={args.model} ===")
        grouped_cv(X, Y, groups, args, screen_w, screen_h, feats)
        leave_session_out(X, Y, sessions, args, screen_w, screen_h, feats)

    # Fit the final model on ALL data with the chosen feature set, and save.
    X = build_X(cache, args.feats)
    scaler = StandardScaler().fit(X)
    model = make_model(args)
    model.fit(scaler.transform(X), Y)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "model": model,
        "scaler": scaler,
        "feats": args.feats,
        "eye_size": eye_size,
        "model_type": args.model,
        "screen_w": screen_w,
        "screen_h": screen_h,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[Train] Saved final {args.feats}/{args.model} model (fit on all {len(Y)} samples) to {args.out}")


if __name__ == "__main__":
    main()
