import os
import pickle
import numpy as np
import pandas as pd

from database import CalibrationDatabase

# The 14-feature Golden vector, in the exact order the app feeds predict().
FEATURE_COLS = [
    "left_ratio_x", "left_ratio_y",
    "right_ratio_x", "right_ratio_y",
    "head_yaw", "head_pitch", "head_roll",
    "head_tx", "head_ty", "head_tz",
    "face_width", "face_height", "face_center_x", "face_center_y",
]


class StandardScaler:
    """
    Zero-mean / unit-variance scaler. Neural nets converge far better when the
    14 raw features (ratios ~0-1, head translations in hundreds, angles in
    degrees) are put on a common scale. Also used on the two screen-coordinate
    targets so the loss is balanced across the X and Y heads.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        # Guard against constant columns producing div-by-zero.
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        return X * self.std_ + self.mean_


class MLP:
    """
    A small multilayer perceptron implemented from scratch in NumPy.

    Architecture: [n_in] -> hidden(ReLU) ... -> [n_out] linear.
    Optimizer:    Adam, hand-rolled.
    Loss:         mean squared error (L2), with weight decay for regularization.
    Training:     mini-batch gradient descent with early stopping on a held-out
                  validation split (important given only ~400 calibration points).

    No autograd, no framework -- every gradient below is derived by hand.
    """

    def __init__(self, layer_sizes, l2=1e-4, seed=42):
        self.layer_sizes = layer_sizes
        self.l2 = l2
        self.rng = np.random.default_rng(seed)

        self.weights = []   # list of (fan_in, fan_out) matrices
        self.biases = []    # list of (fan_out,) vectors

        # Adam moment estimates, one pair per parameter tensor.
        self._mW, self._vW = [], []
        self._mB, self._vB = [], []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization: good for ReLU networks.
            W = self.rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)
            self._mW.append(np.zeros_like(W))
            self._vW.append(np.zeros_like(W))
            self._mB.append(np.zeros_like(b))
            self._vB.append(np.zeros_like(b))

    # ---- forward / backward ------------------------------------------------

    def _forward(self, X):
        """Returns output plus the pre/post activations needed for backprop."""
        activations = [X]      # a[0] = input
        pre_acts = []          # z[l] = a[l-1] @ W + b (before ReLU)
        a = X
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            pre_acts.append(z)
            if i < n_layers - 1:
                a = np.maximum(0.0, z)   # ReLU on hidden layers
            else:
                a = z                    # linear output layer
            activations.append(a)
        return a, activations, pre_acts

    def _backward(self, activations, pre_acts, y_true):
        """Backpropagation. Returns gradients for every weight/bias."""
        n = y_true.shape[0]
        n_layers = len(self.weights)
        gradW = [None] * n_layers
        gradB = [None] * n_layers

        y_pred = activations[-1]
        # d(MSE)/d(output): 2/N * (y_pred - y_true)
        delta = (2.0 / n) * (y_pred - y_true)

        for l in reversed(range(n_layers)):
            a_prev = activations[l]
            gradW[l] = a_prev.T @ delta + self.l2 * self.weights[l]
            gradB[l] = delta.sum(axis=0)
            if l > 0:
                # Propagate through the weights, then through the ReLU of layer l-1.
                delta = delta @ self.weights[l].T
                delta = delta * (pre_acts[l - 1] > 0)   # ReLU derivative
        return gradW, gradB

    def _adam_step(self, gradW, gradB, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for l in range(len(self.weights)):
            # ---- weights ----
            self._mW[l] = beta1 * self._mW[l] + (1 - beta1) * gradW[l]
            self._vW[l] = beta2 * self._vW[l] + (1 - beta2) * (gradW[l] ** 2)
            m_hat = self._mW[l] / (1 - beta1 ** t)
            v_hat = self._vW[l] / (1 - beta2 ** t)
            self.weights[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # ---- biases ----
            self._mB[l] = beta1 * self._mB[l] + (1 - beta1) * gradB[l]
            self._vB[l] = beta2 * self._vB[l] + (1 - beta2) * (gradB[l] ** 2)
            m_hat_b = self._mB[l] / (1 - beta1 ** t)
            v_hat_b = self._vB[l] / (1 - beta2 ** t)
            self.biases[l] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def predict(self, X):
        return self._forward(X)[0]

    def fit(self, X, y, X_val=None, y_val=None, epochs=2000, batch_size=32,
            lr=1e-3, patience=150, verbose=False):
        """Mini-batch Adam with early stopping on validation MSE."""
        n = X.shape[0]
        best_val = np.inf
        best_params = None
        wait = 0
        t = 0  # Adam timestep

        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            Xs, ys = X[perm], y[perm]
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]
                _, acts, pre = self._forward(xb)
                gW, gB = self._backward(acts, pre, yb)
                t += 1
                self._adam_step(gW, gB, lr, t)

            # ---- validation & early stopping ----
            if X_val is not None:
                val_mse = np.mean((self.predict(X_val) - y_val) ** 2)
            else:
                val_mse = np.mean((self.predict(X) - y) ** 2)

            if val_mse < best_val - 1e-9:
                best_val = val_mse
                best_params = ([W.copy() for W in self.weights],
                               [b.copy() for b in self.biases])
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"[MLP] Early stop at epoch {epoch}, best val MSE={best_val:.5f}")
                    break

            if verbose and epoch % 100 == 0:
                print(f"[MLP] epoch {epoch:4d}  val_mse={val_mse:.5f}")

        if best_params is not None:
            self.weights, self.biases = best_params
        return best_val


class GazeNeuralRegressor:
    """
    Drop-in replacement for GazeRegressor backed by a from-scratch NumPy MLP.

    Mirrors the same public interface (train_from_db / save_model / load_model /
    predict) so app.py and calibration.py can swap it in with a one-line change.
    A single network predicts both screen_x and screen_y (2 output units); this
    lets the shared hidden layers learn eye/head features useful for both axes.
    """
    def __init__(self, models_dir="models", filename="gaze_net.pkl",
                 db_dir="data", db_filename="calibration_golden.db",
                 hidden_layers=(64, 32)):
        self.models_dir = models_dir
        self.filepath = os.path.join(models_dir, filename)
        self.db = CalibrationDatabase(db_dir=db_dir, db_filename=db_filename)
        self.hidden_layers = tuple(hidden_layers)

        self.net = None
        self.x_scaler = None
        self.y_scaler = None

        os.makedirs(self.models_dir, exist_ok=True)

    # ---- training ----------------------------------------------------------

    def _prepare_dataframe(self):
        records, columns = self.db.load_latest_session(include_test_pass=False)
        if not records:
            print("[Neural] No calibration records found in SQLite database. Cannot train.")
            return None
        df = pd.DataFrame(records, columns=columns)
        if len(df) < 15:
            print("[Neural] Too few database records in the latest session to train.")
            return None
        return df

    def train_from_db(self, verbose=True):
        """Loads the latest calibration session and fits the MLP on 14 features -> (x, y)."""
        df = self._prepare_dataframe()
        if df is None:
            return False

        X = df[FEATURE_COLS].values.astype(np.float64)
        Y = df[["screen_x", "screen_y"]].values.astype(np.float64)

        # Standardize inputs and targets.
        self.x_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(Y)
        Xs = self.x_scaler.transform(X)
        Ys = self.y_scaler.transform(Y)

        # Deterministic train/validation split (~15% held out for early stopping).
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(Xs))
        n_val = max(4, int(0.15 * len(Xs)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        layer_sizes = [X.shape[1], *self.hidden_layers, 2]
        self.net = MLP(layer_sizes, l2=1e-4, seed=42)
        self.net.fit(
            Xs[tr_idx], Ys[tr_idx],
            X_val=Xs[val_idx], y_val=Ys[val_idx],
            epochs=3000, batch_size=32, lr=1e-3, patience=200, verbose=verbose,
        )

        # Report accuracy in real pixel units on the full session.
        pred = self.y_scaler.inverse_transform(self.net.predict(Xs))
        self._report_metrics(Y, pred)

        self.save_model()
        return True

    def _report_metrics(self, y_true, y_pred):
        def r2(yt, yp):
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        r2x = r2(y_true[:, 0], y_pred[:, 0])
        r2y = r2(y_true[:, 1], y_pred[:, 1])
        px_err = np.sqrt(((y_true - y_pred) ** 2).sum(axis=1))
        print(f"[Neural] MLP trained. X R^2: {r2x:.3f}, Y R^2: {r2y:.3f} | "
              f"mean px error: {px_err.mean():.1f}, median: {np.median(px_err):.1f}")

    # ---- persistence -------------------------------------------------------

    def save_model(self):
        payload = {
            "layer_sizes": self.net.layer_sizes,
            "weights": self.net.weights,
            "biases": self.net.biases,
            "x_mean": self.x_scaler.mean_, "x_std": self.x_scaler.std_,
            "y_mean": self.y_scaler.mean_, "y_std": self.y_scaler.std_,
        }
        with open(self.filepath, "wb") as f:
            pickle.dump(payload, f)
        print(f"[Neural] Saved MLP model to {self.filepath}")

    def load_model(self):
        if not os.path.exists(self.filepath):
            print(f"[Neural] Model file {self.filepath} not found.")
            return False
        with open(self.filepath, "rb") as f:
            p = pickle.load(f)

        self.net = MLP(p["layer_sizes"])
        self.net.weights = p["weights"]
        self.net.biases = p["biases"]
        self.x_scaler = StandardScaler()
        self.x_scaler.mean_, self.x_scaler.std_ = p["x_mean"], p["x_std"]
        self.y_scaler = StandardScaler()
        self.y_scaler.mean_, self.y_scaler.std_ = p["y_mean"], p["y_std"]

        print("[Neural] MLP model loaded successfully.")
        return True

    # ---- inference ---------------------------------------------------------

    def predict(self, left_x, left_y, right_x, right_y, yaw, pitch, roll,
                tx, ty, tz, face_w, face_h, face_cx, face_cy):
        """Predicts (screen_x, screen_y) from the 14-feature vector."""
        if self.net is None:
            raise ValueError("[Neural] Model is not loaded or trained yet.")

        features = np.array([[
            left_x, left_y, right_x, right_y,
            yaw, pitch, roll, tx, ty, tz,
            face_w, face_h, face_cx, face_cy,
        ]], dtype=np.float64)

        Xs = self.x_scaler.transform(features)
        pred = self.y_scaler.inverse_transform(self.net.predict(Xs))[0]

        # Guard against out-of-distribution frames: a ReLU net with a linear
        # output head extrapolates without bound, so a garbage feature vector
        # (blink artifact, lost face) could emit an absurd coordinate that
        # poisons the downstream EMA smoother. The calibrated screen spans
        # roughly mean +/- 2*std of the targets; mean +/- 4*std is a safe
        # envelope that never clips a legitimate prediction.
        lo = self.y_scaler.mean_ - 4.0 * self.y_scaler.std_
        hi = self.y_scaler.mean_ + 4.0 * self.y_scaler.std_
        pred = np.clip(pred, lo, hi)
        return float(pred[0]), float(pred[1])
