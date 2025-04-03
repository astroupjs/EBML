# EBML
EBML is a repository for a paper (not yet published). It includes a dataset of synthetic light curves of eclipsing binaries. You can find the dataset at the following link:

[Dataset of Synthetic Light Curves](https://u.pcloud.link/publink/show?code=kZMm285Zoy7Q3IAQOakIshhv4jTeH8OAtS4y#folder=25535342132&tpl=publicfolderlist)

## Folder Structure and Content

### Main Folder: `EBML`
The `EBML` folder contains the following subfolders:

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


