# LCMS-Net: Deep learning for cause-of-death screening

## Overview

Current preprocessing workflows for untargeted metabolomics using liquid chromatography - high resolution mass spectrometry (LC-HRMS) are often time-consuming, require extensive domain knowledge, lack reproducibility, or fail to detect some metabolites entirely. Here, we present LCMS-Net, an end-to-end deep learning model for the analysis of LC-HRMS data. Our method operates directly on raw data and explicitly models its spatial properties to utilize all available information.

![alt text](https://github.com/lisamenacher/DL_for_CoD_Prediction/blob/main/Fig_1.png)

## System Requirements & Installation

Please make sure that you meet all system requirements before proceeding.

### Hardware 

LCMS-Net was tested on a Linux (RedHat 8) system with a 4 core CPU, 128GB RAM, and a RTX 4070 12GB GPU. Running the code on a standard desktop computer with sufficient RAM is also possible.

### Software

Clone the the GitHub repository and use the environment.yml file to install the needed conda environment for LCMS-Net. All used software versions can also be retrieved from the environment.yml file.

Software versions:
Python 3.12.7
scikit-lean 1.5.1
scikit-optimize 0.10.2
pyOpenMS 3.2
TensorFlow 2.18.1

Use:
```
conda env create --name lcms_net --file=environment.yml
```

## Licence

This project is licensed under the Apache License, Version 2.0 and is open for any academic use. Papers related to this project will be submitted, please cite for use.

Lisa M. Menacher (lisa.menacher@liu.se)

Oleg Sysoev (oleg.sysoev@liu.se)

