#!/usr/bin/env python3
"""
rPPG pipeline: Verkruysse 2008 (Green), CHROM de Haan 2013, POS Wang 2017.
MediaPipe Face Mesh forehead ROI, Butterworth bandpass, Welch HR estimation.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp


# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Research (rPPG)\rPPG emulation"
OUTPUT_DIR = r"C:\Users\Prana\Desktop\rppg_project"

FOREHEAD_LM = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(OUTPUT_DIR, "face_landmarker.task")

BP_LOW    = 0.65   # Hz
BP_HIGH   = 4.0    # Hz
BP_ORDER  = 6

POS_WIN   = 32     # frames
HR_WIN_S  = 10.0   # seconds for temporal HR window
HR_STEP_S = 1.0    # seconds step between windows

VIDEOS = ["vid.avi",          "vid_2.avi",          "vid_3.avi",          "vid_4.avi"]
GTS    = ["ground_truth.txt", "ground_truth_2.txt", "ground_truth_3.txt", "ground_truth_4.txt"]
ALGOS  = ["Green", "CHROM", "POS"]
COLORS = {"Green": "#4CAF50", "CHROM": "#2196F3", "POS": "#F44336"}


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_gt(path):
    """
    Load 3-row ground-truth file.
    Row 0 = PPG signal, row 1 = HR (BPM), row 2 = timestamps (s).
    Returns (ppg, hr_bpm, timestamps) as 1-D float64 arrays.
    """
    data = np.loadtxt(path)
    return data[0], data[1], data[2]


def ensure_model():
    """Download the FaceLandmarker .task model once; returns local path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        print(f"  Downloading face_landmarker.task (~5 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Download complete.")
    return MODEL_PATH


def extract_rgb(video_path):
    """
    Read every frame, run MediaPipe FaceLandmarker (Tasks API, VIDEO mode),
    define the forehead ROI via a convex hull of FOREHEAD_LM landmarks,
    and return per-frame mean RGB.

    Returns
    -------
    rgb        : (N, 3) float64   mean [R, G, B] per detected frame
    fps        : float            video frame-rate reported by OpenCV
    timestamps : (N,) float64     frame timestamps in seconds
    """
    FaceLandmarker        = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model()),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    rgb_list, ts_list = [], []
    frame_idx = 0

    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w      = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts_ms     = int(frame_idx * 1000 / fps)
            result    = detector.detect_for_video(mp_image, ts_ms)

            if result.face_landmarks:
                lm  = result.face_landmarks[0]   # list of NormalizedLandmark
                pts = np.array(
                    [[int(lm[i].x * w), int(lm[i].y * h)]
                     for i in FOREHEAD_LM],
                    dtype=np.int32,
                )
                hull = cv2.convexHull(pts)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [hull], 255)

                roi = frame[mask > 0]          # (K, 3) BGR pixels inside ROI
                if roi.size:
                    rgb_list.append([
                        float(roi[:, 2].mean()),   # R (OpenCV stores BGR)
                        float(roi[:, 1].mean()),   # G
                        float(roi[:, 0].mean()),   # B
                    ])
                    ts_list.append(frame_idx / fps)

            frame_idx += 1

    cap.release()
    return np.asarray(rgb_list, dtype=np.float64), fps, np.asarray(ts_list)


# ── Algorithms ─────────────────────────────────────────────────────────────────

def bandpass(sig, fs):
    """6th-order Butterworth bandpass [BP_LOW, BP_HIGH] Hz with zero-phase filtering."""
    nyq  = fs / 2.0
    b, a = sp.butter(BP_ORDER, [BP_LOW / nyq, BP_HIGH / nyq], btype="band")
    return sp.filtfilt(b, a, sig)


def algo_green(rgb, fps):
    """Verkruysse et al. 2008: bandpass-filtered green channel."""
    return bandpass(rgb[:, 1], fps)


def algo_chrom(rgb, fps):
    """de Haan & Jeanne 2013 CHROM.

    Normalize each channel by its temporal mean, then:
      Xs = 3Rn - 2Gn
      Ys = 1.5Rn + Gn - 1.5Bn
    Bandpass both; combine as S = Xf - alpha*Yf, alpha = std(Xf)/std(Yf).
    """
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Rn = R / (R.mean() + 1e-8)
    Gn = G / (G.mean() + 1e-8)
    Bn = B / (B.mean() + 1e-8)

    Xs = 3.0*Rn - 2.0*Gn
    Ys = 1.5*Rn + Gn - 1.5*Bn

    Xf = bandpass(Xs, fps)
    Yf = bandpass(Ys, fps)

    alpha = Xf.std() / (Yf.std() + 1e-8)
    return Xf - alpha * Yf


def algo_pos(rgb, fps):
    """Wang et al. 2017 POS with overlap-add (step = 1 frame).

    For each window of POS_WIN frames:
      normalize channels by window mean,
      S1 = Gn - Bn,  S2 = -2Rn + Gn + Bn,
      h  = S1 + (std(S1)/std(S2)) * S2   (zero-meaned)
    Overlap-add and divide by overlap count, then bandpass.
    """
    N, L  = len(rgb), POS_WIN
    H     = np.zeros(N, dtype=np.float64)
    cnt   = np.zeros(N, dtype=np.float64)

    total = N - L + 1
    report_every = max(1, total // 10)

    for s in range(total):
        if s % report_every == 0:
            pct = 100 * s / total
            print(f"\r    POS window {s}/{total} ({pct:.0f}%)…", end="", flush=True)

        w  = rgb[s:s+L]
        Rn = w[:, 0] / (w[:, 0].mean() + 1e-8)
        Gn = w[:, 1] / (w[:, 1].mean() + 1e-8)
        Bn = w[:, 2] / (w[:, 2].mean() + 1e-8)

        S1 = Gn - Bn
        S2 = -2.0*Rn + Gn + Bn
        h  = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        h -= h.mean()

        H[s:s+L]   += h
        cnt[s:s+L] += 1

    print(f"\r    POS: {total} windows done.           ")

    nz = cnt > 0
    H[nz] /= cnt[nz]
    return bandpass(H, fps)


# ── HR estimation ──────────────────────────────────────────────────────────────

def welch_hr(sig, fs, nperseg=None):
    """Estimate HR in BPM as the dominant frequency in [BP_LOW, BP_HIGH] Hz."""
    nperseg  = nperseg or min(len(sig), max(int(fs * 10), 64))
    freqs, p = sp.welch(sig, fs=fs, nperseg=nperseg)
    band     = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not band.any():
        return np.nan
    return float(freqs[band][p[band].argmax()] * 60.0)


def hr_over_time(rppg, fps, ts):
    """
    Sliding-window temporal HR estimation.
    Uses HR_WIN_S-second windows (adaptive if signal is short),
    stepped by HR_STEP_S seconds.

    Returns (hr_timestamps, hr_bpm) arrays.
    """
    duration = len(rppg) / fps
    win_s    = min(HR_WIN_S, duration * 0.4)
    win_s    = max(win_s, min(5.0, duration * 0.3))
    win      = max(int(win_s * fps), 32)
    step     = max(1, int(HR_STEP_S * fps))
    nperseg  = min(win, 256)

    hrs, tss = [], []
    for s in range(0, len(rppg) - win + 1, step):
        hrs.append(welch_hr(rppg[s:s+win], fps, nperseg=nperseg))
        tss.append(ts[s + win // 2])

    if not hrs:
        return np.array([]), np.array([])
    return np.asarray(tss), np.asarray(hrs)


def compute_metrics(pred_ts, pred_hr, gt_ts, gt_hr):
    """
    MAE and RMSE of predicted HR vs ground-truth HR.
    GT HR is linearly interpolated onto predicted timestamps;
    only the overlapping time range is used.
    """
    if len(pred_ts) == 0 or len(pred_hr) == 0:
        return np.nan, np.nan

    lo  = max(pred_ts[0],  gt_ts[0])
    hi  = min(pred_ts[-1], gt_ts[-1])
    sel = (pred_ts >= lo) & (pred_ts <= hi)
    if not sel.any():
        return np.nan, np.nan

    pred  = pred_hr[sel]
    ref   = np.interp(pred_ts[sel], gt_ts, gt_hr)
    valid = ~(np.isnan(pred) | np.isnan(ref))
    if not valid.any():
        return np.nan, np.nan

    d = pred[valid] - ref[valid]
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_signals(rppg_dict, vname):
    """Save raw (bandpass-filtered) pulse signals for all algorithms."""
    fig, axes = plt.subplots(len(ALGOS), 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"rPPG Pulse Signals — {vname}", fontsize=13)
    for ax, alg in zip(axes, ALGOS):
        sig, ts = rppg_dict[alg]
        ax.plot(ts, sig, color=COLORS[alg], lw=0.7)
        ax.set_ylabel(alg, fontsize=10)
        ax.set_xlim(ts[0], ts[-1])
        ax.yaxis.set_tick_params(labelsize=8)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"signals_{vname}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_spectra(rppg_dict, fps, vname):
    """Save Welch power spectra (restricted to HR band) for all algorithms."""
    fig, axes = plt.subplots(1, len(ALGOS), figsize=(15, 4))
    fig.suptitle(f"Power Spectra (Welch) — {vname}", fontsize=13)
    for ax, alg in zip(axes, ALGOS):
        sig, _ = rppg_dict[alg]
        nperseg = min(len(sig), int(fps * 10))
        f, p    = sp.welch(sig, fs=fps, nperseg=nperseg)
        band    = (f >= BP_LOW) & (f <= BP_HIGH)
        if band.any():
            peak_hz = f[band][p[band].argmax()]
            ax.fill_between(f[band], p[band], alpha=0.35, color=COLORS[alg])
            ax.plot(f[band], p[band], color=COLORS[alg], lw=1.2)
            ax.axvline(peak_hz, color="k", lw=0.9, ls="--",
                       label=f"{peak_hz*60:.1f} BPM")
            ax.legend(fontsize=9)
        ax.set_title(alg)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"spectra_{vname}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_bar_chart(results, vnames):
    """Save grouped bar chart of MAE and RMSE across videos and algorithms."""
    n = len(results)
    x = np.arange(n)
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, alg in enumerate(ALGOS):
        maes  = [r[alg][0] for r in results]
        rmses = [r[alg][1] for r in results]
        axes[0].bar(x + i*w, maes,  w, label=alg, color=COLORS[alg], edgecolor="white")
        axes[1].bar(x + i*w, rmses, w, label=alg, color=COLORS[alg], edgecolor="white")

    for ax, metric in zip(axes, ["MAE (BPM)", "RMSE (BPM)"]):
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"HR Estimation Error — {metric}", fontsize=12)
        ax.set_xticks(x + w)
        ax.set_xticklabels(vnames, rotation=15, ha="right")
        ax.legend()
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "mae_rmse_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:7.2f}" if not np.isnan(v) else "    N/A"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results, vnames = [], []

    for vid_file, gt_file in zip(VIDEOS, GTS):
        vid_path = os.path.join(DATA_DIR, vid_file)
        gt_path  = os.path.join(DATA_DIR, gt_file)
        vname    = os.path.splitext(vid_file)[0]

        print(f"\n{'='*64}")
        print(f"  {vid_file}")
        print(f"{'='*64}")

        if not os.path.isfile(vid_path):
            print(f"  [SKIP] Missing: {vid_path}")
            continue
        if not os.path.isfile(gt_path):
            print(f"  [SKIP] Missing: {gt_path}")
            continue

        # ── Ground truth ──────────────────────────────────────────
        gt_ppg, gt_hr, gt_ts = load_gt(gt_path)
        print(f"  GT   : {len(gt_hr)} samples, "
              f"mean = {np.nanmean(gt_hr):.1f} BPM, "
              f"t = [{gt_ts[0]:.1f}, {gt_ts[-1]:.1f}] s")

        # ── Extract RGB ───────────────────────────────────────────
        print("  Extracting forehead RGB (MediaPipe)…")
        rgb, fps, ts = extract_rgb(vid_path)
        n_det = len(rgb)
        print(f"  Video: {n_det} face-detected frames, "
              f"fps = {fps:.2f}, duration = {n_det/fps:.1f} s")

        if n_det < POS_WIN + 20:
            print("  [SKIP] Too few frames after face detection.")
            continue

        # ── Run algorithms ────────────────────────────────────────
        algo_fns  = {"Green": algo_green, "CHROM": algo_chrom, "POS": algo_pos}
        rppg_dict = {}
        for alg, fn in algo_fns.items():
            print(f"  [{alg}]", flush=True)
            rppg_dict[alg] = (fn(rgb, fps), ts)

        # ── Compute metrics ───────────────────────────────────────
        row = {}
        print()
        for alg in ALGOS:
            sig, sig_ts    = rppg_dict[alg]
            hr_ts, hr_vals = hr_over_time(sig, fps, sig_ts)
            mae, rmse      = compute_metrics(hr_ts, hr_vals, gt_ts, gt_hr)
            print(f"    {alg:>6}: MAE = {fmt(mae)} BPM   RMSE = {fmt(rmse)} BPM")
            row[alg] = (
                mae  if not np.isnan(mae)  else 999.0,
                rmse if not np.isnan(rmse) else 999.0,
            )

        results.append(row)
        vnames.append(vname)

        # ── Save per-video plots ──────────────────────────────────
        print()
        plot_signals(rppg_dict, vname)
        plot_spectra(rppg_dict, fps, vname)

    # ── Summary table ─────────────────────────────────────────────────────────
    if not results:
        print("\nNo videos processed successfully.")
        return

    COL = 11   # column width per metric value

    print(f"\n{'='*74}")
    print("  RESULTS SUMMARY (BPM)")
    print(f"{'='*74}")
    hdr = f"{'Video':<14}"
    for a in ALGOS:
        hdr += f"  {(a+' MAE'):>{COL}} {(a+' RMSE'):>{COL}}"
    print(hdr)
    print("-" * len(hdr))

    for vname, row in zip(vnames, results):
        line = f"{vname:<14}"
        for a in ALGOS:
            line += f"  {row[a][0]:>{COL}.2f} {row[a][1]:>{COL}.2f}"
        print(line)

    if len(results) > 1:
        print("-" * len(hdr))
        line = f"{'Mean':<14}"
        for a in ALGOS:
            mm = np.mean([r[a][0] for r in results])
            mr = np.mean([r[a][1] for r in results])
            line += f"  {mm:>{COL}.2f} {mr:>{COL}.2f}"
        print(line)

    # ── Save comparison bar chart ──────────────────────────────────────────────
    print()
    plot_bar_chart(results, vnames)
    print("\nAll done.")


if __name__ == "__main__":
    main()
