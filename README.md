# Probing Hidden States in VLMs

This repository contains a Python project for building a COCO-based question dataset, running a vision-language model (VLM) on image-question pairs, extracting hidden states, training diagnostic probes, and evaluating cross-category generalisation. The code is organised around the files in `src/` and the executable scripts in `scripts/`.

The project is based on the files currently present in this repo and reflects the actual directory structure observed in the workspace.

## Project purpose

From the code structure and scripts, the project appears to do the following:

- build a dataset manifest from COCO image annotations
- sample image IDs and construct present/absent/adversarial questions
- run model inference on image-question pairs
- save hidden-state checkpoint files for each row
- evaluate model correctness and confidence
- train a probe classifier on hidden states
- compare within-distribution vs cross-category generalisation
- produce comparison and plotting outputs

## Actual project structure

This is the repository layout as it exists in the current workspace:

```text
ProbingHiddenStatesInVLMs/
├── AGENTS.md
├── CS3043S-2 (1).pdf
├── Report.pdf
├── requirements.txt
├── instances_val2017.json
├── .git/
├── .gitignore
├── data/
│   ├── baseline_comparison.csv
│   ├── checkpoint_d_disjointness.csv
│   ├── disagreements.csv
│   ├── generalization_gap.csv
│   ├── layer_auroc.csv
│   ├── manifest.csv
│   ├── q9_baseline_metrics.csv
│   ├── q9_examples.csv
│   ├── row_metadata.json
│   ├── split_balance.csv
│   └── train_val_split.json
├── checkpoints/
│   ├── row_000000_image_000000569273.safetensors
│   ├── row_000001_image_000000569273.safetensors
│   ├── row_000002_image_000000569273.safetensors
│   ├── ...
│   └── many additional row_*.safetensors checkpoints
├── logs/
│   └── generated runtime logs (if produced in this environment)
├── plots/
│   └── generated plots (if produced in this environment)
├── scripts/
│   ├── build_dataset.py
│   ├── make_plots.py
│   ├── run_generalization.py
│   ├── run_inference.py
│   ├── run_probes.py
│   └── __pycache__/
├── src/
│   ├── COCOSubset.py
│   ├── __init__.py
│   ├── __pycache__/
│   ├── dataset_construction.py
│   ├── inference.py
│   ├── probing.py
│   └── generalization.py
├── tests/
│   └── test_dataset_construction.py
├── val2017/
│   └── COCO validation image data
└── README.md
```

## Core source files

The main implementation is in `src/`:

- `src/dataset_construction.py`
  - creates seeds
  - loads the COCO subset
  - computes category co-occurrence
  - samples image IDs
  - builds the question set
  - saves and loads the manifest

- `src/COCOSubset.py`
  - wraps the COCO dataset access
  - resolves category names and image metadata
  - retrieves present categories for each image

- `src/inference.py`
  - loads the model and processor
  - builds prompts for image-question pairs
  - parses model output into yes/no answers
  - extracts hidden states
  - saves checkpoint files as `.safetensors`

- `src/probing.py`
  - defines the feature pooling logic
  - creates train/validation splits
  - trains a logistic-regression probe
  - evaluates probe accuracy and AUROC
  - validates split balance

- `src/generalization.py`
  - constructs cross-category splits
  - compares probe performance to baseline confidence
  - produces disagreement tables for analysis

## Script entry points

The repository includes runnable scripts in `scripts/`:

- `scripts/build_dataset.py`
  - constructs the manifest from COCO data and writes `data/manifest.csv`

- `scripts/run_inference.py`
  - loads the model, runs inference across the manifest, and saves per-row checkpoints

- `scripts/run_probes.py`
  - builds hidden-state features, trains probes, saves layer metrics, and writes metadata

- `scripts/run_generalization.py`
  - performs the cross-category transfer experiment and produces generalisation outputs

- `scripts/make_plots.py`
  - creates a plot from the layer metrics CSV

## Dependencies

The exact dependencies are declared in `requirements.txt`:

```txt
# Core data libraries
numpy>=1.26,<3
pandas>=2.2,<3
matplotlib>=3.9,<4
tqdm>=4.66,<5

# Machine learning
scikit-learn>=1.5,<2

# PyTorch
torch>=2.5,<3
torchvision>=0.20,<1

# COCO annotations
pycocotools>=2.0.8,<3

# Vision Language Model
transformers>=4.46,<5
safetensors>=0.4.5,<1.0.0
tokenizers>=0.20,<0.22.0

# Image/model utilities
Pillow>=10,<13
accelerate>=1,<2
```

This means the project depends on:

- Python scientific tooling: `numpy`, `pandas`, `matplotlib`, `tqdm`
- ML: `scikit-learn`
- deep learning: `torch`, `torchvision`
- COCO access: `pycocotools`
- VLM support: `transformers`, `safetensors`, `tokenizers`
- image handling: `Pillow`
- inference acceleration: `accelerate`

## Installation

### 1) Clone the repository

```bash
git clone <repo-url>
cd ProbingHiddenStatesInVLMs
```

### 2) Create a virtual environment

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

#### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

#### Windows (Command Prompt)

```cmd
py -3.10 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Recommended environment notes

### GPU support

The project uses `torch` and `transformers`, so a CUDA-capable machine is recommended for faster model inference. You can verify whether PyTorch detects CUDA via:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### CPU fallback

If CUDA is not available, the project still has a CPU path; it just runs slower.

## Data expected by the project

The repo includes the following data-related inputs:

- `instances_val2017.json`
  - COCO annotation file used for the dataset construction logic

- `val2017/`
  - image directory for the validation set

- `data/manifest.csv`
  - manifest created by the dataset builder

- `checkpoints/`
  - generated per-example safetensors files containing hidden-state or inference results

The project also writes intermediate and final outputs into:

- `data/`
- `checkpoints/`
- `logs/`
- `plots/`

## Typical execution flow

From the scripts in the repo, the expected flow is:

1. Build the dataset manifest

```bash
python scripts/build_dataset.py
```

2. Run VLM inference

```bash
python scripts/run_inference.py
```

3. Train and evaluate probes

```bash
python scripts/run_probes.py
```

4. Run generalisation analysis

```bash
python scripts/run_generalization.py
```

5. Generate plots

```bash
python scripts/make_plots.py
```

## Output artifacts

The outputs already present in this repository include files such as:

- `data/manifest.csv`
- `data/layer_auroc.csv`
- `data/baseline_comparison.csv`
- `data/generalization_gap.csv`
- `data/disagreements.csv`
- `data/q9_examples.csv`
- `data/train_val_split.json`
- `data/row_metadata.json`
- `checkpoints/row_*.safetensors`

These files are consistent with the project logic in `src/` and the scripts in `scripts/`.

## Notes on repository conventions

This repo is not using a broad monorepo layout; it is a focused ML/data-analysis project with:

- a Python source package in `src/`
- script runners in `scripts/`
- generated data in `data/`
- saved model checkpoints in `checkpoints/`
- plots and logs in separate output directories

## Practical advice

- Use a fresh virtual environment for this project.
- Install everything from `requirements.txt` before running scripts.
- Keep the repo root as the working directory when executing the scripts.
- If a script fails, first confirm that the expected folder structure exists (`data/`, `val2017/`, `checkpoints/`).

## Summary

This project is a local, assignment-style VLM probing workflow built around:

- COCO annotation processing
- dataset generation
- model inference with hidden-state extraction
- probe training and evaluation
- cross-category evaluation and disagreement analysis

The repository structure above matches the actual files currently present in the workspace and should be used as the authoritative project layout.
