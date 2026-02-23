# 🔍 Template Matching — Classical Computer Vision

> A classical, non-ML computer vision mini-project using **OpenCV**, **NumPy**, and **Matplotlib**.  
> Detects a target object in a scene by sliding a template across the image and scoring each position.

---

## 📌 Overview

This project implements a full **template matching pipeline** from scratch using only classical image processing techniques — no machine learning, no pretrained models.

Given a small **template image** (the object to find) and a larger **scene image**, the algorithm locates all instances of the object and returns their positions with confidence scores.

**Task demonstrated:** Detecting gold coins on a textured tabletop, ignoring silver coin distractors.

---

## 🗂️ Project Structure

```
├── template_matching.py       # Clean Python script (runs in Spyder / any IDE)
├── template_matching.ipynb    # Jupyter Notebook with step-by-step visualisations
├── report.docx                # 1-page assignment report with screenshots
└── README.md
```

---

## ⚙️ Pipeline

```
Input Scene + Template
        │
        ▼
① Grayscale Conversion
        │
        ▼
② cv2.matchTemplate()  →  Response Map (score at every position)
        │
        ▼
③ Threshold + Peak-based NMS  →  Filtered Detections
        │
        ▼
④ Draw Bounding Boxes + Confidence Scores  →  Final Output
```

**Method used:** `cv2.TM_CCOEFF_NORMED` — Normalised Cross-Correlation Coefficient

At each position (x, y) the score measures how well the template's intensity pattern correlates with the scene patch. Score range: **−1** (inverse) → **0** (no match) → **+1** (perfect match).

---

## 📊 Visualisations (Notebook)

The Jupyter notebook renders **8 inline plot sections**:

| Section | Content |
|---------|---------|
| Step 0 | Input scene (with ground-truth markers) + template |
| Step 1 | Colour → grayscale conversion side-by-side |
| Step 2 | Raw score map, normalised heatmap, threshold mask + score histogram |
| Step 3 | Before vs after NMS with peak markers |
| Step 4 | Input vs annotated output |
| Summary | Full 6-panel pipeline overview |
| Threshold Explorer | Results at 4 thresholds (0.80 → 0.40) |
| Zoomed Crops | Template vs each detected patch side-by-side |

---

## 🚀 Getting Started

### Requirements

```bash
pip install opencv-python numpy matplotlib
```

### Run the Script

```bash
python template_matching.py
```

Outputs saved automatically:
- `input_scene.png` — the generated scene
- `input_template.png` — the template
- `output_matches.png` — annotated detections
- `pipeline_summary.png` — full pipeline figure

### Run the Notebook

```bash
jupyter notebook template_matching.ipynb
```

Or open directly in **VS Code**, **JupyterLab**, or **Google Colab**.

### Use Your Own Images

In `template_matching.py`, replace the demo image builder with your own files:

```python
scene    = cv2.imread("your_scene.jpg")
template = cv2.imread("your_template.jpg")
```

---

## 🎛️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `cv2.TM_CCOEFF_NORMED` | Matching method |
| `threshold` | `0.52` | Minimum score to accept a detection |
| `max_det` | `20` | Maximum number of detections (NMS cap) |

**Tip:** Lower the threshold to catch more instances (risk: false positives). Raise it for stricter matching.

Available methods:

| Method | Notes |
|--------|-------|
| `TM_CCOEFF_NORMED` ✅ | Best for general use — lighting invariant |
| `TM_CCORR_NORMED` | Fast, less robust to background |
| `TM_SQDIFF_NORMED` | Good for uniform backgrounds |

---

## ✅ Advantages

1. **Zero training** — works purely on pixel intensities, no labelled data needed
2. **Fast** — FFT-accelerated convolution runs in real time
3. **Interpretable** — confidence scores in \[0, 1\] are directly meaningful

## ⚠️ Limitations

- Sensitive to **scale and rotation** changes in the target object
- Degrades under significant **lighting variation**
- Visually similar distractors can cause **false positives**

## 💡 Proposed Improvements

1. **Multi-scale** — apply template at several scales (0.7× – 1.3×) and keep the best
2. **Multi-angle** — rotate template at discrete steps (e.g. every 10°) before matching
3. **Edge-based template** — pre-process with Canny for lighting robustness
4. **Adaptive threshold** — use `mean + 2×std` of the response map instead of a fixed value

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/) — image processing and template matching
- [NumPy](https://numpy.org/) — array operations
- [Matplotlib](https://matplotlib.org/) — visualisation

---

## 📄 License

This project is for educational purposes as part of a Computer Vision course assignment.
