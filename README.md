# EBML

EBML is a repository for the paper (under review). It includes the trained models and a dataset of synthetic light curves of eclipsing binaries. You can find the storage at the following link:

[Trained Models and Dataset of Synthetic Light Curves](https://u.pcloud.link/publink/show?code=kZMm285Zoy7Q3IAQOakIshhv4jTeH8OAtS4y#folder=25535342132&tpl=publicfolderlist)

---

## 1. Storage Folder Structure and Content

### Main Folder: `EBML`
The `EBML` storage folder contains the following subfolders:

#### Subfolder: `Models`
Contains trained machine learning models for eclipsing binary classification and spot detection. Structure:
- `Models/Gaia/`: Trained ResNet and ViT models for Gaia G passband.
- `Models/OGLE/`: Trained ResNet and ViT models for OGLE I passband.
- `Models/TESS/`: Trained ResNet and ViT models for TESS passband.

**Naming Convention for Model Files:**
`2class_"model_type"_"model_architecture"_"passband"_hexbin.pth`
- **model_type**: Type of classification (e.g., binary, spotted, etc.)
- **model_architecture**: Architecture used (e.g., ResNet, ViT)
- **passband**: Photometric passband (e.g., Gaia, OGLE, TESS)

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

## 2. Repository Structure

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

### Notebooks
Complete workflow from data preparation to model inference and evaluation:
- `metrics_for_classification.ipynb`: Classification metrics for binary models.
- `metrics_for_spot_detection.ipynb`: Metrics for spot detection models.
- `models_learning_progress.ipynb`: Visualize/track model learning progress.
- `models_learning_study.ipynb`: Visualizations and metrics for model training progress.
- `tutorial_to_predict_real_data.ipynb`: Predict class/spot existence for real Gaia light curves.
- `tutorial_to_train_models.ipynb`: Train/evaluate models on custom/extended datasets.
- `tutorial.ipynb`: Step-by-step tutorial for synthetic Gaia light curve data.

---

## 3. Getting Started

### Prerequisites
- Python 3.9 or 3.10 (recommended)
- pip (Python package manager)
- git (for cloning the repository)

### Installation Steps
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd EBML
   ```
2. **(Optional but recommended) Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```
   - If you encounter issues with numpy version compatibility, use:
     ```sh
     pip install "numpy<2"
     ```
4. **(Optional) Install Jupyter for running notebooks:**
   ```sh
   pip install notebook
   ```
5. **Download the required data and pretrained models:**
   - Follow the instructions in the notebooks to download synthetic and real Gaia light curve data, as well as pretrained model files.

### Running the Notebooks
- Open your desired notebook in VS Code or Jupyter:
  ```sh
  jupyter notebook
  # or use VS Code's built-in notebook support
  ```
- Follow the step-by-step instructions in each notebook for data preparation, model training, and prediction.


