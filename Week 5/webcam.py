"""
Webcam Inference — ASL Citizen LSTM (84-dim, new mediapipe tasks API)

UX:
  Press SPACE → 3-second countdown → auto-records for 3 seconds → predicts
  Your hands are completely free while signing!

Controls:
  SPACE — trigger a new sign (during countdown/recording it cancels)
  C     — clear history
  Q     — quit

Usage:
    python3 webcam_inference.py
    python3 webcam_inference.py --model saved_models/asl_citizen_fc_model.pth
    python3 webcam_inference.py --countdown 3 --record 4 --top-k 3
"""

import cv2
import time
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from pathlib import Path
from collections import deque

from asl_citizen_processor import Extractor, FEATURE_DIM
from how2sign.lstm_model import Video_LSTM_morelayers


# ─── states ────────────────────────────────────────────────────────────────────
IDLE       = "idle"
COUNTDOWN  = "countdown"
RECORDING  = "recording"


# ─── model helpers ─────────────────────────────────────────────────────────────

def load_model(model_path, num_classes, hidden_size, n_layers, dropout, feature_dim, device):
    model = Video_LSTM_morelayers(
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=n_layers,
        num_classes=num_classes,
        input_size=feature_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model.to(device)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def draw_landmarks(frame, result):
    """Draw hand skeleton onto frame using pure cv2 — no mediapipe solutions needed."""
    h, w = frame.shape[:2]
    for hand_lms in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 180, 90), 1)


def predict(model, frames, label_to_gloss, top_k, device):
    if len(frames) < 5:
        return [("(too short)", 0.0)]
    video = torch.stack(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(video), dim=1)[0]
    k = min(top_k, len(label_to_gloss))
    top_probs, top_idx = probs.topk(k)
    return [(label_to_gloss[i.item()], p.item()) for i, p in zip(top_idx, top_probs)]


# ─── UI ────────────────────────────────────────────────────────────────────────

def draw_ui(frame, state, results, history, elapsed, countdown_sec, record_sec):
    h, w = frame.shape[:2]

    # ── top status bar ────────────────────────────────────────────────────────
    if state == COUNTDOWN:
        remaining = max(0.0, countdown_sec - elapsed)
        bar_color = (0, 140, 200)
        msg       = f"Get ready …  {remaining:.1f}s"
    elif state == RECORDING:
        remaining = max(0.0, record_sec - elapsed)
        bar_color = (0, 0, 200)
        msg       = f"● SIGNING  {remaining:.1f}s remaining"
    else:
        bar_color = (30, 30, 30)
        msg       = "SPACE = new sign   C = clear   Q = quit"

    cv2.rectangle(frame, (0, 0), (w, 54), bar_color, -1)
    cv2.putText(frame, msg, (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── countdown big number ──────────────────────────────────────────────────
    if state == COUNTDOWN:
        remaining = max(0.0, countdown_sec - elapsed)
        big       = str(int(remaining) + 1)
        (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)
        cx = (w - tw) // 2
        cy = (h + th) // 2 - 40
        cv2.putText(frame, big, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 8)

    # ── recording progress bar ────────────────────────────────────────────────
    if state == RECORDING:
        frac    = min(1.0, elapsed / record_sec)
        bar_end = int(frac * w)
        cv2.rectangle(frame, (0, 54), (bar_end, 60), (0, 80, 255), -1)

    # ── predictions ───────────────────────────────────────────────────────────
    if results:
        panel_top = h - 170
        cv2.rectangle(frame, (0, panel_top), (w, h), (15, 15, 15), -1)
        cv2.putText(frame, "Prediction:", (15, panel_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        for i, (gloss, prob) in enumerate(results):
            y = panel_top + 52 + i * 38
            cv2.rectangle(frame, (15, y - 18), (15 + int(prob * 280), y + 5),
                          (0, 210, 100) if i == 0 else (50, 110, 70), -1)
            cv2.putText(frame, f"{gloss}  {prob*100:.1f}%", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75 if i == 0 else 0.58,
                        (255, 255, 255), 2 if i == 0 else 1)

    # ── recent history ────────────────────────────────────────────────────────
    if history:
        cv2.putText(frame, "Recent:", (w - 230, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
        for i, word in enumerate(reversed(list(history))):
            cv2.putText(frame, word, (w - 230, 103 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 190, 255), 1)

    return frame


# ─── main ──────────────────────────────────────────────────────────────────────

def run_webcam(
    model_path:    str   = "saved_models/asl_citizen_fc_model.pth",
    processed_dir: str   = "asl_citizen_processed",
    top_k:         int   = 3,
    hidden_size:   int   = 150,
    n_layers:      int   = 7,
    dropout:       float = 0.5,
    countdown_sec: int   = 3,
    record_sec:    int   = 3,
):
    processed_path = Path(processed_dir)

    # ── label map ─────────────────────────────────────────────────────────────
    label_map_path = processed_path / "label_map.csv"
    if not label_map_path.exists():
        print("❌ label_map.csv not found — run asl_citizen_processor.py first")
        return
    label_map      = pd.read_csv(label_map_path)
    label_to_gloss = dict(zip(label_map["label"].astype(int), label_map["gloss"]))
    num_classes    = len(label_to_gloss)

    # ── config ────────────────────────────────────────────────────────────────
    config_path = processed_path / "config.csv"
    feature_dim = (int(pd.read_csv(config_path).iloc[0]["feature_dim"])
                   if config_path.exists() else FEATURE_DIM)
    print(f"Classes: {num_classes}   Feature dim: {feature_dim}")

    # ── model ─────────────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path} — run train_asl_citizen.py first")
        return
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))
    print(f"Device: {device}")
    model = load_model(model_path, num_classes, hidden_size,
                       n_layers, dropout, feature_dim, device)

    # ── webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print(f"\n🎥 Ready!")
    print(f"   Press SPACE → {countdown_sec}s countdown → {record_sec}s recording → prediction")
    print(f"   Q to quit\n")

    state           = IDLE
    state_start     = 0.0
    recorded_frames = []
    results         = []
    history         = deque(maxlen=8)

    with Extractor() as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            display = frame.copy()
            now     = time.time()
            elapsed = now - state_start

            # ── always extract for landmark drawing ──────────────────────────
            _, live_result = extractor.extract_with_result(frame)                              if state != RECORDING else (None, None)

            # ── state transitions ─────────────────────────────────────────────
            if state == COUNTDOWN and elapsed >= countdown_sec:
                state           = RECORDING
                state_start     = now
                elapsed         = 0.0
                recorded_frames = []
                extractor.reset_timestamp()
                print("● Signing now!")

            elif state == RECORDING:
                t, last_result = extractor.extract_with_result(frame)
                if t is not None:
                    recorded_frames.append(t.cpu())
                if last_result is not None:
                    draw_landmarks(display, last_result)

                if elapsed >= record_sec:
                    print(f"  {len(recorded_frames)} frames — predicting …")
                    results = predict(model, recorded_frames,
                                      label_to_gloss, top_k, device)
                    history.append(results[0][0])
                    for g, p in results:
                        print(f"  {'→' if g == results[0][0] else ' '} "
                              f"{g}  ({p*100:.1f}%)")
                    state = IDLE

            # ── draw landmarks (always visible) ──────────────────────────────
            if state != RECORDING and live_result is not None:
                draw_landmarks(display, live_result)

            # ── draw UI ───────────────────────────────────────────────────────
            draw_ui(display, state, results, list(history),
                    elapsed, countdown_sec, record_sec)
            cv2.imshow("ASL Citizen — Sign Recognition", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if state == IDLE:
                    state       = COUNTDOWN
                    state_start = now
                    results     = []
                    print(f"Get ready … ({countdown_sec}s)")
                else:
                    # cancel mid-countdown or mid-recording
                    state = IDLE
                    print("Cancelled.")

            elif key == ord("c"):
                history.clear()
                results = []
                print("Cleared.")

            elif key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="saved_models/asl_citizen_fc_model.pth")
    p.add_argument("--processed-dir", default="asl_citizen_processed")
    p.add_argument("--top-k",         type=int,   default=3)
    p.add_argument("--hidden-size",   type=int,   default=150)
    p.add_argument("--layers",        type=int,   default=7)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--countdown",     type=int,   default=3,
                   help="Seconds between SPACE press and recording start")
    p.add_argument("--record",        type=int,   default=3,
                   help="Seconds of recording per sign")
    args = p.parse_args()

    run_webcam(
        model_path=args.model,
        processed_dir=args.processed_dir,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        n_layers=args.layers,
        dropout=args.dropout,
        countdown_sec=args.countdown,
        record_sec=args.record,
    )