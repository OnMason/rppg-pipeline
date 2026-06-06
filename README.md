# Remote Photoplethysmography (rPPG) Heart Rate Estimation

A research-oriented Python implementation of three classical remote photoplethysmography (rPPG) algorithms for contactless heart rate estimation from facial videos.

This project uses MediaPipe Face Mesh to detect facial landmarks, extracts a forehead region of interest (ROI), computes RGB color traces, and estimates heart rate using signal processing techniques. The performance of each algorithm is evaluated against ground-truth physiological measurements.

## Algorithms Implemented

### 1. Green Channel Method
Based on:
> Verkruysse, Svaasand, & Nelson (2008)

Uses the green color channel as the primary source of pulsatile blood volume information and applies bandpass filtering to recover the cardiac signal.

### 2. CHROM
Based on:
> de Haan & Jeanne (2013)

Projects normalized RGB signals into chrominance subspaces to improve robustness against motion and illumination changes.

### 3. POS (Plane-Orthogonal-to-Skin)
Based on:
> Wang et al. (2017)

Uses a skin-tone orthogonal projection and overlap-add reconstruction to extract pulse signals from facial videos.

---

## Features

- Automatic face detection and tracking using MediaPipe Face Mesh
- Forehead ROI extraction using facial landmarks
- Mean RGB signal extraction per frame
- Butterworth bandpass filtering
- Welch power spectral density heart rate estimation
- Temporal heart rate tracking
- Ground-truth HR comparison
- MAE and RMSE evaluation metrics
- Signal visualization
- Frequency spectrum analysis
- Cross-video benchmarking

---

## Pipeline

```text
Video Input
     ↓
Face Detection (MediaPipe)
     ↓
Forehead ROI Extraction
     ↓
RGB Signal Extraction
     ↓
Green / CHROM / POS
     ↓
Bandpass Filtering
     ↓
Welch PSD Analysis
     ↓
Heart Rate Estimation
     ↓
MAE / RMSE Evaluation
```

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Matplotlib

---

## Project Structure

```text
.
├── videos/
│   ├── vid.avi
│   ├── vid_2.avi
│   ├── vid_3.avi
│   └── vid_4.avi
│
├── ground_truth/
│   ├── ground_truth.txt
│   ├── ground_truth_2.txt
│   ├── ground_truth_3.txt
│   └── ground_truth_4.txt
│
├── outputs/
│   ├── signals_*.png
│   ├── spectra_*.png
│   └── mae_rmse_comparison.png
│
└── rppg_pipeline.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/rppg-heart-rate-estimation.git
cd rppg-heart-rate-estimation
```

Install dependencies:

```bash
pip install opencv-python mediapipe numpy scipy matplotlib
```

---

## Usage

Update the paths in the configuration section:

```python
DATA_DIR = "path_to_videos"
OUTPUT_DIR = "path_to_output_folder"
```

Run:

```bash
python rppg_pipeline.py
```

The program will:

1. Process each video
2. Extract forehead RGB signals
3. Run Green, CHROM, and POS algorithms
4. Estimate heart rate
5. Compare predictions to ground truth
6. Generate plots and evaluation metrics

---

## Evaluation Metrics

Performance is measured using:

### Mean Absolute Error (MAE)

```text
MAE = mean(|Predicted HR - Ground Truth HR|)
```

### Root Mean Square Error (RMSE)

```text
RMSE = sqrt(mean((Predicted HR - Ground Truth HR)^2))
```

Lower values indicate better heart rate estimation accuracy.

---

## Example Output

The pipeline generates:

### Pulse Signal Visualization
- Bandpass-filtered rPPG signals for each algorithm

### Frequency Spectra
- Welch PSD plots
- Dominant frequency detection
- Estimated BPM peaks

### Performance Comparison
- MAE comparison across videos
- RMSE comparison across videos
- Aggregate benchmarking plots

---

## Applications

- Contactless heart-rate monitoring
- Biomedical signal processing
- Healthcare AI
- Telemedicine
- Human-computer interaction
- Physiological computing
- Computer vision research

---

## References

### Verkruysse et al. (2008)
Verkruysse, W., Svaasand, L. O., & Nelson, J. S.
"Remote plethysmographic imaging using ambient light."

### de Haan & Jeanne (2013)
de Haan, G., & Jeanne, V.
"Robust pulse rate from chrominance-based rPPG."

### Wang et al. (2017)
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G.
"Algorithmic principles of remote PPG."

---

## Future Improvements

- Deep-learning-based rPPG models
- Real-time webcam processing
- Motion artifact reduction
- Multi-person heart rate estimation
- SpO₂ estimation
- Blood pressure estimation research

---

## License

This project is intended for research and educational purposes.
