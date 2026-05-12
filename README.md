# tb-plasma-emission

Source code accompanying the manuscript **"Tuberculosis detection from plasma-emission images of exhaled breath using convolutional neural networks"** (Gündel, Helfer & Molz; submitted to *Biomedical Signal Processing and Control*).

This repository contains the full pipeline used to preprocess plasma-emission images from the *BDEE-Device*, train and evaluate classical and deep-learning classifiers, and produce the interpretability and domain-shift analyses reported in the paper.

---

## Pipeline overview

```
scripts/
├── 01_build_registry.py         # Index images per patient/session, deduplication
├── 02_exploratory_analysis.py   # EDA: class distribution, intensity stats, mean image
├── 03_preprocess.py             # Quality filtering, ROI extraction, resizing, augmentation
├── 04_run_baseline.py           # Handcrafted features + Random Forest / SVM
├── 05_run_cnn.py                # ResNet-50, EfficientNet-B0, MobileNetV2 (transfer learning)
├── 06_gradcam_analysis.py       # Grad-CAM interpretability for the selected model
├── 07_domain_shift.py           # Cross-dataset evaluation (PEVA <-> PESC)
├── 08_generate_figures.py       # Paper-ready figures from logged results
└── 09_blob_crop_experiment.py   # Blob-cropping ablation to probe shortcut learning
```

Reusable modules live in `src/`:

```
src/
├── data/         # Dataset registry, patient-aware splitter, preprocessing, augmentation
├── features/     # Handcrafted feature extractors (HOG, LBP, GLCM, Hu moments, etc.)
├── models/       # Baseline classifiers + CNN architectures + training loop
├── evaluation/   # Metrics, Grad-CAM, plotting helpers
└── utils/        # Reproducibility (seeding, deterministic ops)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10. GPU training was performed on an NVIDIA RTX 3060 (6 GB VRAM); CPU inference is feasible for the MobileNetV2 model.

---

## Reproducing the experiments

The pipeline is designed to be run in order. Configure paths and hyperparameters in `configs/experiment_config.yaml` before starting.

```bash
python scripts/01_build_registry.py
python scripts/02_exploratory_analysis.py
python scripts/03_preprocess.py
python scripts/04_run_baseline.py
python scripts/05_run_cnn.py
python scripts/06_gradcam_analysis.py
python scripts/07_domain_shift.py
python scripts/09_blob_crop_experiment.py
python scripts/08_generate_figures.py
```

All training scripts use deterministic seeding (`src/utils/reproducibility.py`) and patient-level cross-validation (`src/data/patient_splitter.py`) to avoid information leakage.

---

## Data availability

The plasma-emission image datasets (PEVA, PESC) are **not** included in this repository. Data were collected from incarcerated participants under a research protocol approved by the UNISC Research Ethics Committee (CAAE 78659024.0.0000.5343) and by the Penitentiary Service of Rio Grande do Sul (SUSEPE, ruling No. 31/2024), which restricts secondary use.

De-identified data may be made available by the corresponding author upon reasonable request, subject to a data-use agreement and additional ethical approval.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Gundel2026PlasmaTB,
  title   = {Tuberculosis detection from plasma-emission images of exhaled breath using convolutional neural networks},
  author  = {Gündel, Mateus Elias and Helfer, Gilson Augusto and Molz, Rolf Fredi},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Acknowledgements

This work was supported by FAPERGS, CAPES, CNPq (Process 88881.710388/2022-01 — CAPES PDPG Consolidação 13407), and the Graduate Program in Industrial Systems and Processes (PPGSPI) of the University of Santa Cruz do Sul (UNISC).

## Contact

Mateus Elias Gündel — gundel5@mx2.unisc.br
