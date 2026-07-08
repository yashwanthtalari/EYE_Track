"""
Simple image data collector for the VisionPoint eye-tracking project.

Displays a full-screen grid of target dots. For each dot you look at and hold
your gaze on, the script captures ~N face-crop images from the webcam and labels
each one with the on-screen target coordinate. The result is a folder of images
plus a labels.csv that an image-based gaze model can be trained on directly.

Usage:
    python collect_data.py                 # 5x5 grid, 30 images per point
    python collect_data.py --grid 3 3      # 3x3 grid
    python collect_data.py --samples 60    # 60 images per point
    python collect_data.py --out dataset   # custom output folder

Controls (in the full-screen window):
    SPACE : start capturing the current highlighted point
    S     : skip the current point
    ESC   : quit and save whatever has been collected so far
"""
import cv2
import numpy as np
import argparse
import csv
import os
import time
import pyautogui

from camera import VideoCamera
from detector import FaceMeshDetector


def parse_args():
    p = argparse.ArgumentParser(description="Collect labeled face-crop images for gaze training.")
    p.add_argument("--out", default="dataset", help="Output folder (default: dataset)")
    p.add_argument("--grid", type=int, nargs=2, default=[5, 5],
                   metavar=("COLS", "ROWS"), help="Grid columns and rows (default: 5 5)")
    p.add_argument("--samples", type=int, default=30, help="Images to capture per point (default: 30)")
    p.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    p.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    p.add_argument("--margin", type=float, default=0.08,
                   help="Fraction of the screen kept as a border, so dots aren't in the extreme corners (default: 0.08)")
    p.add_argument("--crop-pad", type=float, default=0.4,
                   help="Extra padding around the detected face as a fraction of face size (default: 0.4)")
    p.add_argument("--crop-size", type=int, default=224,
                   help="Output face-crop resolution in pixels, square (default: 224)")
    # --- Tier D: continuous / multi-head-position collection ---
    p.add_argument("--mode", choices=["grid", "continuous"], default="grid",
                   help="grid = dwell on each dot (classic); continuous = follow a roaming dot (default: grid)")
    p.add_argument("--rounds", type=int, default=3,
                   help="continuous: number of head-position rounds (variety for head invariance) (default: 3)")
    p.add_argument("--round-seconds", type=float, default=40.0,
                   help="continuous: seconds of capture per round (default: 40)")
    p.add_argument("--capture-every", type=int, default=2,
                   help="continuous: save every Nth detected frame (default: 2)")
    return p.parse_args()


def build_grid_points(cols, rows, margin):
    """Return normalized (x, y) targets on a cols x rows grid, in row-major order."""
    points = []
    lo, hi = margin, 1.0 - margin
    for r in range(rows):
        # Snake ordering: alternate row direction so consecutive points are near
        # each other, minimizing large eye jumps between captures.
        col_range = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        ny = lo if rows == 1 else lo + (hi - lo) * r / (rows - 1)
        for c in col_range:
            nx = lo if cols == 1 else lo + (hi - lo) * c / (cols - 1)
            points.append((nx, ny))
    return points


def crop_face(bgr_frame, landmarks, frame_w, frame_h, pad, out_size):
    """
    Crop a square region around the detected face and resize to out_size.
    landmarks are normalized [0,1]; returns None if no face / degenerate box.
    """
    if landmarks is None:
        return None

    xs = landmarks[:, 0] * frame_w
    ys = landmarks[:, 1] * frame_h
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    fw = max_x - min_x
    fh = max_y - min_y
    if fw <= 1 or fh <= 1:
        return None

    # Use a square box (side = larger face dimension + padding) centred on the face.
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    side = max(fw, fh) * (1.0 + pad)
    half = side / 2.0

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    # Clamp to frame bounds.
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(frame_w, x2), min(frame_h, y2)
    if x2c - x1c < 2 or y2c - y1c < 2:
        return None

    crop = bgr_frame[y1c:y2c, x1c:x2c]

    # Pad back to a square if the box ran off an edge, so faces near the frame
    # border aren't stretched by the resize.
    ch, cw = crop.shape[:2]
    side_px = max(ch, cw)
    canvas = np.zeros((side_px, side_px, 3), dtype=np.uint8)
    oy = (side_px - ch) // 2
    ox = (side_px - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop

    return cv2.resize(canvas, (out_size, out_size))


def draw_target(ui, point, screen_w, screen_h, color):
    px = int(point[0] * screen_w)
    py = int(point[1] * screen_h)
    cv2.circle(ui, (px, py), 26, color, 2)
    cv2.circle(ui, (px, py), 8, color, -1)
    return px, py


# --------------------------------------------------------------------------
# Tier D: continuous capture (roaming dot) with multiple head positions.
# A dot sweeps smoothly across the whole screen while frames are captured
# continuously and labeled with the dot's live position. Repeated over several
# "rounds", each with a different head pose, so the model learns to separate
# head movement from eye movement -- the main cause of poor cross-session error.
# --------------------------------------------------------------------------
HEAD_PROMPTS = [
    "Sit naturally  -  head CENTERED",
    "Turn/lean your head slightly LEFT",
    "Turn/lean your head slightly RIGHT",
    "Move a little CLOSER to the screen",
    "Lean back  -  a little FARTHER away",
    "Tilt your head slightly (chin up/down)",
]


def build_sweep_path(margin, rows=7):
    """Serpentine raster of normalized waypoints covering the whole screen."""
    lo, hi = margin, 1.0 - margin
    pts = []
    for r in range(rows):
        y = lo + (hi - lo) * r / (rows - 1)
        xs = (lo, hi) if r % 2 == 0 else (hi, lo)
        for x in xs:
            pts.append((x, y))
    return pts


def sample_path(waypoints, prog):
    """Position along the polyline at progress prog in [0,1] (linear interp)."""
    prog = min(max(prog, 0.0), 1.0)
    n = len(waypoints)
    if n == 1:
        return waypoints[0]
    t = prog * (n - 1)
    i = int(t)
    if i >= n - 1:
        return waypoints[-1]
    frac = t - i
    x = waypoints[i][0] + (waypoints[i + 1][0] - waypoints[i][0]) * frac
    y = waypoints[i][1] + (waypoints[i + 1][1] - waypoints[i][1]) * frac
    return (x, y)


def grid_cell_index(nx, ny, cols, rows, margin):
    """Bucket a continuous target into a coarse grid cell, so group-aware
    (leave-point-out) evaluation still works on continuous data."""
    lo, hi = margin, 1.0 - margin
    span = (hi - lo) if hi > lo else 1.0
    fx = (nx - lo) / span
    fy = (ny - lo) / span
    cx = min(cols - 1, max(0, int(fx * cols)))
    cy = min(rows - 1, max(0, int(fy * rows)))
    return cy * cols + cx


def run_continuous(args, camera, detector, writer, images_dir, session_id,
                   screen_w, screen_h, window_name):
    """Capture rounds of a roaming dot at varied head positions. Returns
    (total_saved, aborted)."""
    cols, rows = args.grid
    waypoints = build_sweep_path(args.margin)
    total_saved = 0

    for r in range(args.rounds):
        prompt = HEAD_PROMPTS[r % len(HEAD_PROMPTS)]

        # --- Ready screen: wait for SPACE (ESC aborts) ---
        ready = False
        while not ready:
            success, bgr, rgb = camera.read_frame()
            if not success:
                continue
            face_ok = detector.find_face_landmarks(rgb) is not None
            ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.putText(ui, f"ROUND {r + 1} / {args.rounds}", (60, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.putText(ui, prompt, (60, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ui, "Follow the dot with your EYES (let your head stay in this pose).",
                        (60, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(ui, "SPACE = start round    |    ESC = quit and save",
                        (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
            if not face_ok:
                cv2.putText(ui, "NO FACE DETECTED - align yourself in the camera",
                            (60, screen_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 165, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, ui)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return total_saved, True
            if key == 32 and face_ok:
                ready = True

        # --- Countdown 3..2..1 so the first frames aren't garbage ---
        for c in (3, 2, 1):
            t0 = time.time()
            while time.time() - t0 < 0.7:
                success, bgr, rgb = camera.read_frame()
                ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.putText(ui, str(c), (screen_w // 2 - 30, screen_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6, cv2.LINE_AA)
                cv2.imshow(window_name, ui)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return total_saved, True

        # --- Capture loop: dot roams for round_seconds ---
        start = time.time()
        frame_i = 0
        round_saved = 0
        while True:
            elapsed = time.time() - start
            if elapsed >= args.round_seconds:
                break
            success, bgr, rgb = camera.read_frame()
            if not success:
                continue

            prog = elapsed / args.round_seconds
            tx, ty = sample_path(waypoints, prog)
            landmarks = detector.find_face_landmarks(rgb)
            face_ok = landmarks is not None

            ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            gx, gy = int(tx * screen_w), int(ty * screen_h)
            cv2.circle(ui, (gx, gy), 28, (0, 0, 255), 2)
            cv2.circle(ui, (gx, gy), 9, (0, 0, 255), -1)
            cv2.rectangle(ui, (60, screen_h - 60),
                          (60 + int((screen_w - 120) * prog), screen_h - 45),
                          (0, 200, 255), -1)
            cv2.putText(ui, f"Round {r + 1}/{args.rounds}  -  {prompt}  -  saved {round_saved}",
                        (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            if not face_ok:
                cv2.putText(ui, "NO FACE", (60, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 165, 255), 2, cv2.LINE_AA)

            frame_i += 1
            if face_ok and (frame_i % args.capture_every == 0):
                crop = crop_face(bgr, landmarks, camera.target_width, camera.target_height,
                                 args.crop_pad, args.crop_size)
                if crop is not None:
                    pidx = grid_cell_index(tx, ty, cols, rows, args.margin)
                    fname = f"{session_id}_r{r:02d}_f{frame_i:05d}.jpg"
                    cv2.imwrite(os.path.join(images_dir, fname), crop)
                    writer.writerow([
                        fname, session_id, pidx,
                        f"{tx:.5f}", f"{ty:.5f}",
                        gx, gy, screen_w, screen_h,
                    ])
                    total_saved += 1
                    round_saved += 1

            cv2.imshow(window_name, ui)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return total_saved, True

        print(f"[Collect] Round {r + 1}/{args.rounds} done ({round_saved} images).")

    return total_saved, False


def main():
    args = parse_args()
    cols, rows = args.grid

    os.makedirs(args.out, exist_ok=True)
    images_dir = os.path.join(args.out, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_path = os.path.join(args.out, "labels.csv")

    screen_w, screen_h = pyautogui.size()
    points = build_grid_points(cols, rows, args.margin)

    camera = VideoCamera(device_index=args.device, target_width=args.width, target_height=args.height)
    detector = FaceMeshDetector()

    session_id = f"session_{int(time.time())}"

    # Open labels.csv in append mode so multiple sessions accumulate into one dataset.
    new_file = not os.path.exists(labels_path) or os.path.getsize(labels_path) == 0
    csv_file = open(labels_path, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow([
            "image", "session_id", "point_index",
            "target_norm_x", "target_norm_y",
            "target_px_x", "target_px_y",
            "screen_w", "screen_h",
        ])

    window_name = "VisionPoint - Data Collection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"[Collect] Saving to: {os.path.abspath(args.out)}")

    # --- Tier D: continuous roaming-dot mode with head-position rounds ---
    if args.mode == "continuous":
        print(f"[Collect] Continuous mode: {args.rounds} rounds x {args.round_seconds:.0f}s, "
              f"saving every {args.capture_every} frames.")
        print("[Collect] SPACE = start each round | ESC = quit and save")
        total_saved, aborted = run_continuous(
            args, camera, detector, writer, images_dir, session_id,
            screen_w, screen_h, window_name)
        csv_file.close()
        camera.release()
        cv2.destroyAllWindows()
        tag = "Aborted" if aborted else "Complete"
        print(f"[Collect] {tag}. {total_saved} images written to {images_dir}")
        print(f"[Collect] Labels: {labels_path}")
        return

    print(f"[Collect] {cols}x{rows} grid = {len(points)} points, {args.samples} images each.")
    print("[Collect] SPACE = capture point | S = skip | ESC = quit")

    point_idx = 0
    state = "idle"          # idle -> capturing
    captured = 0
    total_saved = 0
    aborted = False

    while point_idx < len(points):
        success, bgr_frame, rgb_frame = camera.read_frame()
        if not success:
            continue

        landmarks = detector.find_face_landmarks(rgb_frame)
        face_ok = landmarks is not None

        ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

        # Draw all remaining points faintly, current one highlighted.
        for i, pt in enumerate(points):
            if i == point_idx:
                continue
            faint = (60, 60, 60) if i > point_idx else (0, 90, 0)
            gx = int(pt[0] * screen_w)
            gy = int(pt[1] * screen_h)
            cv2.circle(ui, (gx, gy), 10, faint, 1)

        if state == "idle":
            color = (0, 0, 255) if face_ok else (0, 0, 120)   # red (dim if no face)
        else:
            color = (255, 0, 0)                               # blue while capturing
        target_px_x, target_px_y = draw_target(ui, points[point_idx], screen_w, screen_h, color)

        # Header / status text.
        if state == "idle":
            line1 = f"Look at the RED dot, then press SPACE.  (Point {point_idx + 1} of {len(points)})"
            line2 = "S = skip this point   |   ESC = quit and save"
        else:
            line1 = f"Capturing... keep looking at the dot.  ({captured} / {args.samples})"
            line2 = "Hold steady."
        cv2.putText(ui, line1, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ui, line2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        if not face_ok:
            cv2.putText(ui, "NO FACE DETECTED - align yourself in the camera", (50, screen_h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

        # Capture logic.
        if state == "capturing" and face_ok:
            crop = crop_face(bgr_frame, landmarks, camera.target_width, camera.target_height,
                             args.crop_pad, args.crop_size)
            if crop is not None:
                fname = f"{session_id}_p{point_idx:03d}_s{captured:03d}.jpg"
                cv2.imwrite(os.path.join(images_dir, fname), crop)
                writer.writerow([
                    fname, session_id, point_idx,
                    f"{points[point_idx][0]:.5f}", f"{points[point_idx][1]:.5f}",
                    target_px_x, target_px_y,
                    screen_w, screen_h,
                ])
                captured += 1
                total_saved += 1

                if captured >= args.samples:
                    print(f"[Collect] Point {point_idx + 1}/{len(points)} done ({captured} images).")
                    point_idx += 1
                    state = "idle"
                    captured = 0
                    time.sleep(0.3)

        cv2.imshow(window_name, ui)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC
            aborted = True
            break
        elif key == 32:          # SPACE
            if state == "idle" and face_ok:
                state = "capturing"
                captured = 0
        elif key in (ord('s'), ord('S')):
            print(f"[Collect] Skipped point {point_idx + 1}/{len(points)}.")
            point_idx += 1
            state = "idle"
            captured = 0

    csv_file.close()
    camera.release()
    cv2.destroyAllWindows()

    if aborted:
        print(f"[Collect] Aborted. {total_saved} images saved so far.")
    else:
        print(f"[Collect] Complete. {total_saved} images written to {images_dir}")
    print(f"[Collect] Labels: {labels_path}")


if __name__ == "__main__":
    main()
