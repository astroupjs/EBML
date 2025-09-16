# EBML

EBML is a repository for the [paper](https://ui.adsabs.harvard.edu/abs/2025A%26C....5300998P/abstract). It includes trained models and a dataset of synthetic light curves of eclipsing binaries.

## 1. Getting Started

### Requirements
- Python version: 3.9 to 3.12 (due to PyTorch compatibility)
- Required packages: See `requirements.txt`

### Installation
1. Clone the repository:
```bash
git clone https://github.com/astroupjs/EBML.git
cd EBML
```

2. Create and activate a virtual environment (recommended for most users):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

**Conda users:**

If you prefer to use conda, follow these steps:

1. Create and activate a conda environment with a compatible Python version (3.9–3.12):
   ```bash
   conda create -n ebml_env python=3.12
   conda activate ebml_env
   ```
2. Install dependencies using pip inside the conda environment:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For GPU support, install PyTorch via conda:
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```
   *(Replace `cpuonly` with the appropriate CUDA version if you have a compatible GPU)*

## 2. Repository Structure
```
EBML/
├── data/                    # Data files and datasets
├── models/                  # Trained model files (.pth)
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/               # Python scripts and utilities
└── requirements.txt       # Package dependencies
```

### Data Files
The `data` folder contains CSV files with classification results for eclipsing binary systems. Subfolders for training and validation datasets are created by the notebooks.
- **classification_OGLE.csv**: OGLE binaries, with model results for OGLE I and Gaia G.
- **classification_DEBcat.csv**: DEBcat systems, with model results for TESS and Gaia, spot detection, and literature info.
- **classification_WUMaCat.csv**: W UMa catalog systems, with model results for TESS and Gaia, and spot info.

### Models Directory
The `models` directory contains all trained model files for classification and spot detection. Download from storage. Includes a `progress` subfolder for training logs and metadata (JSON files tracking loss, accuracy, etc.).

### Scripts
The `scripts` folder contains Python scripts for generating images, training, and applying ML models:
- **binary_metrics.py**: Binary classification metrics, ROC, reliability diagrams.
- **classify_pytorch_resnet.py**: Load/apply ResNet model to classify a light curve image.
- **classify_pytorch_vit.py**: Load/apply ViT model to classify a light curve image.
- **clean_ogle_csv.py**: Clean OGLE CSV files (remove spaces from columns/values).
- **make_polar_hexbin_images.py**: Generate synthetic polar hexbin images from light curves.
- **model_pytorch_rasnet.py**: Train ResNet model for binary classification.
- **model_pytorch_vit.py**: Train ViT model for binary classification.
- **spot_metrics.py**: Metrics for spot detection (probability histograms, calibration curves).

## 3. Storage Folder Structure

[Trained Models and Dataset of Synthetic Light Curves](https://u.pcloud.link/publink/show?code=kZMm285Zoy7Q3IAQOakIshhv4jTeH8OAtS4y#folder=25535342132&tpl=publicfolderlist)

### Main Folder: `EBML`
The storage contains the following subfolders:

#### Subfolder: `Models`
Contains trained machine learning models for eclipsing binary classification and spot detection:
- `Models/Gaia/`: Trained ResNet and ViT models for Gaia G passband
- `Models/OGLE/`: Trained ResNet and ViT models for OGLE I passband
- `Models/TESS/`: Trained ResNet and ViT models for TESS passband

**Naming Convention for Model Files:**
`2class_"model_type"_"model_architecture"_"passband"_hexbin.pth`
- **model_type**: Type of classification (binary, spotted, etc.)
- **model_architecture**: Architecture used (ResNet, ViT)
- **passband**: Photometric passband (Gaia, OGLE, TESS)

Example files:
- `2class_binary_ResNet_Gaia_hexbin.pth`
- `2class_detachspot_ViT_TESS_hexbin.pth`

#### Subfolder: `Synthetic_LC`
Synthetic light curves generated for eclipsing binary systems. Each CSV file contains 100 flux points per light curve, normalized to the maximum value. Naming: `TypeOfSystem_Spotnes_Passband`.
- **System Types**: Detached (1,000,000), Overcontact (500,000)
- **Passbands**: gaia, I, tess, B, R, V, g, i, kepler

#### Subfolder: `OGLE_LC_binned`
Zipped CSV files with OGLE catalog light curves, binned to 100 points per period.

#### Subfolder: `SelectedLC`
Light curves as CSV tables, standard phase-folded images, and polar+hexbin images. Structure:
- `SelectedLC/GAIA_DEB/`, `SelectedLC/GAIA_OGLE/`, `SelectedLC/GAIA_WUMaCat/`, `SelectedLC/TESS_DEB/`, `SelectedLC/TESS_WUMaCat/`
- Each contains: `Images/`, `Images_hex/`, `LC/`

**CSV Table columns:** TimeG, FG, e_FG, Phase, norm_FG, e_norm_FG

---

### Notebooks
Complete workflow from data preparation to model inference and evaluation:
- `metrics_for_classification.ipynb`: Classification metrics for binary models.
- `metrics_for_spot_detection.ipynb`: Metrics for spot detection models.
- `models_learning_progress.ipynb`: Visualize/track model learning progress.
- `tutorial_to_predict_real_data.ipynb`: Predict class/spot existence for real Gaia light curves.
- `tutorial_to_train_models.ipynb`: Train/evaluate models on custom/extended datasets.

## 4. Using the Notebooks

You can start using the provided Jupyter notebooks immediately. Open any notebook in the `notebooks` folder and follow the step-by-step instructions inside to run analyses, train models, or make predictions. Each notebook includes detailed guidance for its workflow.


