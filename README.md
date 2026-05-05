# Real-time and Resource-Efficient Lane Detection for Autonomous Vehicles in VANET-Enabled Environments using Neuromorphic Computing

**B.Tech Honours Project — Indian Institute of Information Technology Kottayam**  
**Student:** Siraparapu Sai Satya Pavan Charan (Roll No. 2023BCS0078)  
**Supervisor:** Dr. Goutam Mali  
**Department:** Computer Science and Engineering  
**Year:** April 2026

---

## Overview

This project designs, implements, and evaluates a complete neuromorphic lane detection framework for autonomous vehicles operating in Vehicular Ad-hoc Networks (VANETs). Two parallel architectures are compared under identical conditions:

| | CNN Baseline | SNN Proposed |
|--|--|--|
| **Architecture** | U-Net Encoder-Decoder | SpikingLaneNet (LIF neurons) |
| **Computation** | Dense — every pixel, every frame | Sparse — event-driven spikes only |
| **Parameters** | 7,849,601 | 95,073 (~82× fewer) |
| **Model Size** | ~31.4 MB | ~0.4 MB |
| **Complexity** | O(n²k²) dense | O(S) sparse |
| **Energy** | ~50 W (GPU) | ~0.1–1 W (MCU-compatible) |

---

## Repository Structure

```
project/
│
├── Neuromorphic_Lane_Detection_VANET.ipynb     # Full training pipeline notebook
├── Neuromorphic_Lane_Detection_Review.ipynb    # Review presentation notebook (no training)
├── split_bdd100k_10k.py                        # Dataset preparation script
├── filter_bdd100k_labels.py                    # Legacy label filtering script
├── README.md                                   # This file
│
└── outputs/  (generated on Google Drive)
    ├── P01_cnn_pipeline.png
    ├── P02_cnn_histograms.png
    ├── P03_cnn_batch.png
    ├── P04_snn_spike_encoding.png
    ├── P05_cnn_vs_snn_preprocessing.png
    ├── P06_spike_param_tuning.png
    ├── P07_spike_default_vs_tuned.png
    ├── P08_cnn_weather.png
    ├── P09_snn_weather.png
    ├── P10_cnn_vs_snn_weather.png
    ├── P11_weather_spike_analysis.png
    ├── P12_layer_params.png
    ├── P13_param_storage_comparison.png
    ├── cnn_best.pth                            # Best CNN checkpoint
    ├── snn_best.pth                            # Best SNN checkpoint
    ├── comparison_table.csv
    ├── weather_robustness.csv
    └── results.json
```

---

## Dataset Setup

### Step 1 — Download BDD100K 100K

Download the full BDD100K 100K images and labels from the [official BDD100K website](https://bdd-data.berkeley.edu/). You need:
- `bdd100k_images_100k.zip` — 100K driving images
- `bdd100k_labels_release.zip` — per-image JSON annotation files

Your local folder structure should look like:
```
bdd100k_images_100k/
  100k/
    train/    (70,000 images)
    val/      (10,000 images)
    test/     (20,000 images)

bdd100k_labels/
  100k/
    train/    (70,000 .json label files)
    val/      (10,000 .json label files)
    test/     (20,000 .json label files)
```

### Step 2 — Create the 10K Split

Run `split_bdd100k_10k.py` on your local machine. This script randomly samples 10,000 aligned image-label pairs from the full 100K dataset and creates a new split.

```bash
python split_bdd100k_10k.py \
    --images_root "C:/Downloads/bdd100k_images_100k/100k" \
    --labels_root "C:/Downloads/bdd100k_labels/100k" \
    --output_dir  "C:/Downloads/bdd100k_10k_split"
```

**Output structure:**
```
bdd100k_10k_split/
  images/
    train/    7,000 images
    val/      1,000 images
    test/     2,000 images
  labels/
    train/    7,000 .json files  (one per image, same filename stem)
    val/      1,000 .json files
    test/     2,000 .json files
```

**Split sizes:** 7,000 train / 1,000 val / 2,000 test  
**Random seed:** 42 (reproducible)

### Step 3 — Upload to Google Drive

Upload the entire `bdd100k_10k_split/` folder to `MyDrive/`. The notebooks expect:
```
MyDrive/
  bdd100k_10k_split/
    images/  train/ val/ test/
    labels/  train/ val/ test/
    outputs/                     ← auto-created by notebook
```

---

## Quick Start

### For Review Presentation (no training required)

Open `Neuromorphic_Lane_Detection_Review.ipynb` in Google Colab.

1. Runtime → Change runtime type → **T4 GPU**
2. Run **Cell 1** (install packages)
3. Run **Cell 4** (mount Google Drive)
4. Run all remaining cells in order

This notebook covers:
- CNN preprocessing pipeline with visual step-by-step
- SNN spike encoding pipeline with parameter tuning
- CNN vs SNN preprocessing under 5 weather conditions
- Parameter count and disk storage comparison
- 13 publication-quality figures saved to Drive

### For Full Training Pipeline

Open `Neuromorphic_Lane_Detection_VANET.ipynb` in Google Colab.

1. Runtime → Change runtime type → **T4 GPU**
2. Run all cells in order

---

## Notebooks

### `Neuromorphic_Lane_Detection_Review.ipynb`

Focused presentation notebook — **no model training**. Designed for review/seminar presentation.

| Section | Content |
|---------|---------|
| 1 | Environment setup |
| 2 | Dataset loading and verification |
| 3 | CNN preprocessing (Gaussian → CLAHE → ROI → Normalise) |
| 4 | SNN preprocessing (temporal difference → spike encoding) |
| 5 | SNN spike parameter tuning (grid search over θ, T, noise_std) |
| 6 | CNN vs SNN preprocessing under weather conditions (clear/rain/fog/night/shadow) |
| 7 | CNN vs SNN parameter count and disk storage comparison |
| 8 | Summary of all results and generated figures |

### `Neuromorphic_Lane_Detection_VANET.ipynb`

Full research pipeline notebook.

| Section | Content |
|---------|---------|
| 1–3 | Setup, imports, configuration |
| 4 | Drive mount and dataset verification |
| 5–6 | Preprocessing and dataloader |
| 7–8 | CNN U-Net architecture and loss functions |
| 9–10 | CNN training loop and results |
| 11–12 | SNN spike encoding and SpikingLaneNet architecture |
| 12C | **Optuna hyperparameter tuning** (30 trials, TPE Bayesian search) |
| 12D | SNN retraining with best Optuna parameters |
| 13–14 | Lane post-processing and polynomial fitting |
| 15–16 | Evaluation (IoU, Precision, Recall, F1, latency, FPS) and weather robustness |
| 17–18 | Qualitative visualisation and comparative analysis |
| 19–20 | VANET suitability scoring and final summary |

---

## Architecture

### CNN U-Net Baseline

```
Input (3, 256, 512)
  Encoder: [32 → 64 → 128 → 256 ch] + MaxPool2d
  Bottleneck: 512 ch + Dropout(0.2)
  Decoder: Bilinear upsample + skip concat [256 → 128 → 64 → 32 ch]
  Output: Conv(1ch) → Sigmoid → lane probability map
```

- Parameters: **7,849,601**
- Computation: **O(n²k²)** — dense, every pixel, every frame
- Training: Adam + CosineAnnealingLR + Combined BCE-Dice loss

### SNN SpikingLaneNet (Proposed)

```
Spike train input (T-1, B, 1, H, W)
  Encoder: LIFConv(1→16) → LIFConv(16→32, s=2) → LIFConv(32→64, s=2)
  Bottleneck: LIFConv(64→64)
  Decoder: Upsample + skip concat → LIFConv x2 [64→32→16 ch]
  Output: Conv(1ch) → rate coding → sigmoid
```

- Parameters: **95,073** (~82× fewer than CNN)
- Computation: **O(S)** — sparse, event-driven, only active spikes
- Training: BPTT + fast sigmoid surrogate gradient + Optuna hyperparameter tuning

---

## Spike Encoding

Static BDD100K images are converted to pseudo-temporal sequences and encoded as binary spike trains:

```
ΔI_t(x,y) = I_t(x,y) - I_{t-1}(x,y)

S_t(x,y) = 1  if |ΔI_t(x,y)| > θ
           0  otherwise
```

**Tuned parameters (grid search):**

| Parameter | Default | After Tuning | Effect |
|-----------|---------|-------------|--------|
| θ (spike_thresh) | 0.15 | 0.07 | Reduces spike rate from 28.5% → ~7% |
| T (time steps) | 4 | 4 | Temporal integration window |
| noise_std | 0.05 | 0.04 | Inter-frame variation strength |

**Result:** ~93% of pixels remain silent per frame → ~93% fewer MACs than CNN

---

## Optuna Hyperparameter Tuning

The SNN with default parameters achieved Val IoU ≈ 0.0096 due to neuron saturation (28.5% spike rate). Optuna TPE Bayesian search over 30 trials resolves this.

**Search space:**

| Hyperparameter | Range |
|---|---|
| Learning rate η | 1e-4 → 5e-3 |
| Membrane decay β | 0.80 → 0.99 |
| Spike threshold θ | 0.03 → 0.20 |
| Surrogate slope | 5 → 50 |
| Time steps T | 3 → 8 |
| noise_std | 0.02 → 0.15 |
| pos_weight | 5 → 25 |
| Batch size | {2, 4, 8} |

---

## Results (CNN)

| Metric | Value |
|--------|-------|
| Best Val IoU | **0.2820** (epoch 19 of 24) |
| Parameters | 7,849,601 |
| Model size | ~31.4 MB |
| Estimated MACs/frame | ~8.85 M (dense) |

SNN results pending Optuna retraining completion.

---

## Weather Robustness

Both models evaluated under 5 simulated conditions:

| Condition | Simulation |
|-----------|-----------|
| Clear | Identity (baseline) |
| Rain | Gaussian noise + brightness reduction |
| Fog | Contrast reduction + brightness lift |
| Night | 82% luminance reduction |
| Shadow | Left third at 30% brightness |

The SNN's temporal difference encoding is inherently more robust to fog/night because it responds to intensity *changes* rather than absolute values.

---

## VANET Suitability

Weighted suitability score:

```
S_VANET = 0.30 × S_latency + 0.25 × S_accuracy + 0.25 × S_energy
        + 0.10 × S_params  + 0.10 × S_deployability
```

| Requirement | CNN U-Net | SNN SpikingLaneNet |
|-------------|-----------|-------------------|
| Latency < 33ms | Borderline (GPU req.) | Achievable on MCU |
| Power < 5W | ~50W (GPU) | ~0.1–1W |
| 24/7 operation | Thermal throttle risk | Event-driven idle |
| Fits L2 cache | No (~31.4 MB) | Yes (~0.4 MB) |
| V2V data size | ~131 KB (pixel mask) | ~384 bits (polynomial) |

Lane polynomial representation for VANET broadcast: **x = ay² + by + c** (341× data reduction vs pixel mask)

---

## Requirements

```
Python >= 3.10
torch >= 2.0
snntorch >= 0.9
albumentations >= 1.3.0
opencv-python-headless
numpy
pandas
seaborn
matplotlib
scipy
scikit-learn
optuna
```

All packages are installed automatically by Cell 1 in each notebook.

---

## Hardware

All experiments conducted on:
- **Platform:** Google Colab
- **GPU:** Tesla T4 (15.6 GB VRAM)
- **CPU:** 2× vCPU
- **RAM:** ~12 GB
- **Storage:** Google Drive (mounted)

---

## References

1. Yu et al. (2020). BDD100K: A Diverse Driving Dataset. *CVPR 2020.*
2. Zhu et al. (2024). Autonomous Driving with Spiking Neural Networks. *NeurIPS 2024.*
3. Zhou et al. (2023). Computational event-driven vision sensors for in-sensor SNNs. *Nature Electronics.*
4. Eshraghian et al. (2021). Training Spiking Neural Networks Using Lessons From Deep Learning. *arXiv.*
5. Almalag & Weigle (2010). Using Traffic Flow for Cluster Formation in VANETs. *IEEE On-MOVE.*
6. Vodopivec et al. (2012). A Survey on Clustering Algorithms for VANETs. *IEEE TSP.*
7. Ngo et al. (2023). Cooperative Perception With V2V Communication. *IEEE Trans. Veh. Tech.*
8. Ghori et al. (2018). Vehicular Ad-hoc Network (VANET): Review. *IEEE ICIRD.*

---

## Citation

If you use this work, please cite:

```
Siraparapu, S. S. P. C. (2026). Real-time and Resource-Efficient Lane Detection
for Autonomous Vehicles in VANET-Enabled Environments using Neuromorphic Computing.
B.Tech Honours Project, Indian Institute of Information Technology Kottayam.
Supervisor: Dr. Goutam Mali.
```

---

*Indian Institute of Information Technology Kottayam — Department of Computer Science and Engineering — April 2026*
