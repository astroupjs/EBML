# EBML
EBML is a repository for the paper (under review). It includes the trained models and a dataset of synthetic light curves of eclipsing binaries. You can find the storage at the following link:

[Trained Models and Dataset of Synthetic Light Curves](https://u.pcloud.link/publink/show?code=kZMm285Zoy7Q3IAQOakIshhv4jTeH8OAtS4y#folder=25535342132&tpl=publicfolderlist)

## Folder Structure and Content

### Main Folder: `EBML`
The `EBML` folder contains the following subfolders:

#### Subfolder: `Models`
This folder contains trained machine learning models for eclipsing binary classification and spot detection. It has the following structure:
- `Models/Gaia/`: Contains trained ResNet and ViT models for Gaia G passband.
- `Models/OGLE/`: Contains trained ResNet and ViT models for OGLE I passband.
- `Models/TESS/`: Contains trained ResNet and ViT models for TESS passband.
#### Naming Convention for Model Files

The trained model files follow this naming convention:

`2class_"model_type"_"model_architecture"_"passband"_hexbin.pth`

- **model_type**: Specifies the type of classification (e.g., binary or spotted, and for what type of system).
- **model_architecture**: Indicates the architecture used (e.g., ResNet or ViT).
- **passband**: Refers to the photometric passband (e.g., Gaia, OGLE, or TESS).

For example:
- `2class_binary_ResNet_Gaia_hexbin.pth`
- `2class_detachspot_ViT_TESS_hexbin.pth`

#### Subfolder: `Synthetic_LC`
This folder includes synthetic light curves generated for eclipsing binary systems. Each light curve is stored as a CSV file, where each row represents 100 flux points equally distributed by orbital phase and normalized to the maximum value. The files follow the naming convention: `TypeOfSystem_Spotness_Passband`.

- **System Types**:
    - Detached systems: 1,000,000 samples
    - Overcontact systems: 500,000 samples

- **Passbands**:
    - **gaia**: Gaia G
    - **I**: OGLE I
    - **tess**: TESS

#### Subfolder: `OGLE_LC_binned`
This folder contains light curves for eclipsing binary systems from the OGLE catalog. These light curves are binned to 100 points equally distributed over the orbital period.

### Data Files

The `data` folder contains several CSV files with classification results for eclipsing binary systems:

- **classification_OGLE.csv**  
  Contains classification results for OGLE eclipsing binaries. Columns include the object name, Gaia ID, original OGLE classification, and results from various machine learning models (ResNet and ViT) for both binary and spotted classifications in different passbands (OGLE I and Gaia G).

- **classification_DEBcat.csv**  
  Contains classification results for systems from the DEBcat catalog. Columns include the system name, Gaia ID, classification results from different models and passbands (TESS and Gaia), as well as results of visual spot detection and literature information about the presence of spots and bibliographic references.

- **classification_WUMaCat.csv**  
  Contains classification results for systems from the W UMa catalog. Columns include the system name, Gaia ID, classification results from different models and passbands (TESS and Gaia), and information about the presence of spots.

### Scripts

The `sripts` folder contains Python scripts used for generating images, training, and applying machine learning models to classify eclipsing binary light curves:

- **make_syntetic_images_overcontact_hexbin.py**  
  Generates synthetic polar hexbin images from light curve data for overcontact binary systems. It adds noise and outliers to the data, creates images in polar coordinates, and saves them for use in machine learning.

- **model_pytorch_RasNet.py**  
  Trains a ResNet-based convolutional neural network for binary classification of light curve images using PyTorch. Handles data loading, augmentation, training, validation, and model saving.

- **model_pytorch_ViT.py**  
  Trains a Vision Transformer (ViT) model for binary classification of light curve images using PyTorch and the `timm` library. Includes data loading, augmentation, training, validation, and model saving.

- **classify_pytorch_resnet.py**  
  Loads a trained ResNet model and applies it to classify a single light curve image. Outputs the predicted class and probabilities.

- **classify_pytorch_Vit.py**  
  Loads a trained Vision Transformer (ViT) model and applies it to classify a single light curve image. Outputs the predicted class and probabilities.


