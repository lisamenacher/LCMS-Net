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
- Python 3.12.7
- scikit-lean 1.5.1
- scikit-optimize 0.10.2
- imbalanced-learn 0.12.3
- pyOpenMS 3.2
- TensorFlow 2.18.1
- keras-tuner 1.4.7
- keras 3.6.0
(additionally install numpy, pandas, matplotlib, & openpyxl)

Use the following command to install the provided conda environment (should take less than 10 minutes):
```
conda env create --name lcms_net --file=environment.yml
```

## Instructions

To use LCMS-Net the raw LC-HRMS data first needs to be binned using `bin_data.py`. Afterwards, a model instance can be trained using `training.py` and evaluated using `evaluation.py`. Note, that the cause-of-death screening data can not be deposited due to ethical restrictions. However, the following instructions can be applied to other datasets and example data for a demo is provided below. 

### 1. Data binning

For the data binning run the following command:

```
python bin_data.py -meta ../example/meta.xlsx -data ../example/raw_data -save ../save_dir -mode
```

Command line arguments:

- `-meta`: Specify the path to the meta data of the raw LC-HRMS data (as .xlsx-file)
- `-data`: Specify the path to the raw LC-HRMS data. The filenames of the raw data must match the index column of the meta-data file (e.g., the meta data for "Sample_1.mzML" will have the index "Sample_1")
- `-save`: Specify the path where the binned data will be stored
- `-mode`: Specify the binning mode (either "adaptive" or "default")  

### 2. Training 

To train LCMS-Net run the following command:

```
python training.py -meta ../example/meta.xlsx -data ../example/binned_data -save ../save_dir -handleImbalance -norm -aug -lr -batch -epoch
```

Command line arguments:

- `-meta`: Specify the path to the meta data of the raw LC-HRMS data (as .xlsx-file)
- `-data`: Specify the path to the binned LC-HRMS data
- `-handleImbalance`: Specify a mode to address class imbalnce (either "ros", "class_weights" or None)
- `norm`: Boolean value to specify if normalization will be applied
- `aug`: Boolean value to specify if augmentation will be applied
- `lr`: Float value to specify the starting learning rate
- `batch`: Int value to specify the batch size
- `epoch`: Int value to specify the max. number of epochs
- `-save`: Specify the path where the results will be stored

### 3. Evaluation


For the evaluation run the following command:

```
python evaluation.py -meta ../example/meta.xlsx -data ../example/binned_data -save ../save_dir -models ../results/model_dir -test_samples -mode -eval
```

Command line arguments:

- `-meta`: Specify the path to the meta data of the raw LC-HRMS data (as .xlsx-file)
- `-data`: Specify the path to the raw LC-HRMS data
- `-test_samples`: Specify the path to a list of samples of the meta data
- `-models`: Specify the path to the saved model
- `-save`: Specify the path where the results will be stored
- `-mode`: Specify an model mode (either "ensemble" or "single")  
- `-eval`: Specify an evaluation mode (either "eval", "best_reject" or "default_with_reject")  

## Demo

To train LCMS-Net with the provided example data (randomly sampled from two normal distributions), run the following code (should take approx. 5 min):

```
python ./LCMS-Net/training.py
```

For evaluation of a already trained model, run the following code (should take approx. 1 min):
```
python ./LCMS-Net/evaluation.py -eval default
```
(Note you can also chose "default_with_reject" or "best_reject" as value for the variable `eval` to run different types of evaluations)

The example data should given a test accuracy of 100%.

## Licence

This project is licensed under the Apache License, Version 2.0 and is open for any academic use. Papers related to this project will be submitted, please cite for use.

Lisa M. Menacher (lisa.menacher@liu.se)

Oleg Sysoev (oleg.sysoev@liu.se)

