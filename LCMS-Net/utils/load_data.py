import pandas as pd
import numpy as np
import pyopenms as oms
import os

from tensorflow.keras.utils import to_categorical
from itertools import groupby

  
def data_gen(raw_folderpaths, meta_filepath, samples, normalize=True, augment=False):
    """ Data Generator for model training with binned LC-MS. Only samples within provided list will be used.

    Parameters: \\
    raw_folderpaths (list):     List of folderpaths that contain raw LC-MS data \\    
    meta_filepath (str):        Filepath of meta data (xlsx) \\
    samples (list):             List of sample identifiers \\      
    normalize (bool):           Specify if normalization is used \\
    augment (bool):             Specify if augmentation is used 

    Returns: \\
    np.array:                   Binned LC-MS samples
    np.array:                   Class label (as categorical)
    """


    # load meta data (class labels, PMI etc.)
    data = pd.read_excel(meta_filepath.decode(), index_col=0)
    data["labels_num"]= np.unique(data["Group"], return_inverse=True)[1]
    num_classes = len(np.unique(data["Group"]))

    for sample in samples:
        if not np.issubdtype(type(sample), np.integer):
            sample = sample.decode()

        # get sample info
        sample_info = data.loc[sample]

        # get sample from respective folder
        for folder in raw_folderpaths:
            folder = folder.decode()

            filepaths = os.listdir(folder)
            if  str(sample) + ".npy" in filepaths: 
                filepath = folder + "/" + str(sample) + ".npy"
                    
                intensity_matrix = np.load(filepath)
                label = to_categorical(sample_info["labels_num"], num_classes)

                if augment:
                    intensity_matrix = __apply_augmentation(intensity_matrix)

                if normalize:
                    intensity_matrix = __apply_normalization(intensity_matrix)

                yield intensity_matrix, label



def process_data(meta_filepath: str, raw_folderpaths: list, save_path = None, num_classes = 5):
    """Loads all raw LC-MS datafiles (binned or not) from a list of folderpaths and processes it for CNN.

    Parameters: \\
    meta_filepath (str):    Filepath of meta data (xlsx) \\
    raw_folderpaths (list): List of folderpaths that contain raw LC-MS data \\    
    save_path (str):       Path to store binned matrices. If None, data will not be stored. \\    
    num_classes (int):      Number of classes 
    
    Return:
    """

    # load meta data (class labels, PMI etc.)
    data = pd.read_excel(meta_filepath)

    # load each data point and process to equally spaced matrix if not already processed
    for folder in raw_folderpaths:

        # retrieve all files in specified folder and filter out already processed samples
        filepaths = os.listdir(folder)

        if save_path is not None:
            processed_files = os.listdir(save_path)
            processed_files = [file.split(".")[0] for file in processed_files]
            filepaths = [file for file in filepaths if (file.split(".")[0] not in processed_files) & (file.split(".")[0] in data.index)]

        # retrieve data if file-format matches mzdata.xml or npy
        for file in filepaths:

            if file.endswith(".mzML"):

                # get sample information
                if file.split(".")[0] in data.index.tolist(): 

                    filepath = folder + "/" + file
                    exp = oms.MSExperiment()

                    try:
                        oms.MzMLFile().load(filepath, exp)    
                        intensity_matrix = __adaptive_binning(exp, save_path=save_path)
                    except:
                        print(f"ERROR: File {file} could not be loaded!")

            elif file.endswith(".mzdata.xml"): 

                # get sample information
                sample = file.split('.')[0]
                sample_info = data.loc[sample]
                filepath = folder + "/" + file
                exp = oms.MSExperiment()

                try:
                    oms.MzDataFile().load(filepath, exp)    
                    intensity_matrix = __adaptive_binning(exp, save_path=save_path)
                except:
                    print(f"ERROR: File {file} could not be loaded!")


            elif file.endswith(".npy"):

                # get sample information
                sample = file.split('.')[0]
                sample_info = data.loc[sample]
                filepath = folder + "/" + file

                intensity_matrix = np.load(filepath)



def __apply_augmentation(intensity_matrix):

    matrix_shape = intensity_matrix.shape

    rt_shift = np.random.random_integers(-10, 10, 1)[0]
    if rt_shift > 0:
        intensity_matrix = np.delete(np.r_[np.zeros([rt_shift, matrix_shape[1]]), intensity_matrix], [range(matrix_shape[0], (rt_shift + matrix_shape[0]))], axis=0)
    elif rt_shift < 0:
        intensity_matrix = np.delete(np.r_[intensity_matrix, np.zeros([-rt_shift, matrix_shape[1]])], [range(0, -rt_shift)], axis=0)

    return intensity_matrix


def __apply_normalization(intensity_matrix):
    row_max = np.max(intensity_matrix, axis=0, keepdims=True)
    row_max[row_max == 0] = 1
    intensity_matrix = np.divide(intensity_matrix, row_max)

    return intensity_matrix


def __adaptive_binning(raw_LCMS, dim=(856, 1024), rt_start=45, rt_end=660, mz_start=50, mz_end=1000, save_path=None, filename=None):

    # create bins with fixed size for RT based on sampling interval of measurement instrument (about 0.66 to 0.72 seconds)
    rt_bins = pd.interval_range(rt_start-0.72 , rt_end+0.72, freq=0.72) 

    # create adaptive bins for m/z values based on percentage of expected peaks within a bucket of 50 ppm (percentages from Liam's processed data)
    mz_bins = []
    mz_bins_densities = [0.012, 0.048, 0.1, 0.128, 0.136, 0.099, 0.05, 0.049, 0.048, 0.065, 0.026, 0.009, 0.006, 0.045, 0.111, 0.063, 0.001, 0.001, 0.004] # CoD screening
    # mz_bins_densities = [0.01, 0.1, 0.048, 0.14, 0.1, 0.048, 0.048, 0.048, 0.25, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.002, 0.002, 0.002] # prostate

    for i in range(0, int((mz_end-mz_start)/50)):
        if i == 0:
            mz_bins.extend(np.linspace(mz_start, mz_start+50, int(np.round((dim[1]+len(mz_bins_densities)-2)*mz_bins_densities[i],0))))
        else:
            mz_bins.extend(np.linspace(mz_bins[-1], mz_bins[-1]+50, int(np.round((dim[1]+len(mz_bins_densities)-2)*mz_bins_densities[i],0)))[1:])

    if len(mz_bins) != dim[1]:
        mz_bins = mz_bins[:dim[1]]

    # create 0-matrix + helpers
    binned_matrix = np.zeros((len(rt_bins), dim[1]))  

    # iterate over individual spectra and bin data across m/z axis, remove spectra outside of defined RT range
    for i in range(0, raw_LCMS.getNrSpectra()):

        if (raw_LCMS[i].getMSLevel() == 1) and (raw_LCMS[i].getRT() > rt_start and raw_LCMS[i].getRT() < rt_end):
            mz, intensity = raw_LCMS[i].get_peaks()

            # bin m/z values according to previously defined bins + remove values outside of range
            mz_binned = np.digitize(mz, mz_bins) - 1 
            nan_idx = np.argwhere(mz_binned == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)
            grouped_intensity = [np.max([c for _, c in g]) for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])]
            grouped_bins = np.unique(mz_binned)

            # bin RT values according to previously defined bins
            rt_binned = pd.cut([raw_LCMS[i].getRT()], rt_bins)
            rt_binned = rt_binned.rename_categories(list(range(0, len(rt_bins))))
            
            # add binned intensities to data matrix
            if np.sum(binned_matrix[rt_binned[0], :]) != 0:
                curr_val = binned_matrix[rt_binned[0],:]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity

                binned_matrix[rt_binned[0],:] = np.max(np.stack((curr_val, new_val)), axis=0)
            else:
                binned_matrix[rt_binned[0], grouped_bins] = grouped_intensity

    if save_path is not None and os.path.exists(save_path):
        if filename is not None:
            np.save(save_path + '/' + str(filename) + '.npy', binned_matrix)
        else:
            # np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy', binned_matrix) # for mzML 
            np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy', binned_matrix) # for mzData.xml

    return binned_matrix



def __default_binning(raw_LCMS, dim=(1024, 1024), rt_start=45, rt_end=660, mz_start=49.9, mz_end=1000, save_path=None, filename=None):
    """ Bin raw LCMS data according to the specified parameters.

    Parameters: \\
    raw_LCMS (oms.Experiment):  Raw LCMS data \\
    dim (tuple):                Expected shape of binned data \\    
    rt_start (int):             Min. included retention time \\
    rt_end (int):               Max. included retention time\\
    mz_start (int):             Min. included m/z value \\
    mz_end (int):               Max. included m/z value \\
    save_path (None or str):     If None, binned matrix will not be saved. If str, save as .npy file at specifiedPath
    
    Return:
    np.array:                   Binned LCMS data
    """

    # create bins
    mz_bins = pd.interval_range(mz_start, mz_end, periods=dim[1])
    start = [pd.Interval(left=0, right=mz_bins[0].right)]
    end = [pd.Interval(left=mz_bins[-1].left, right=np.Inf)]
    mz_bins = pd.IntervalIndex(start + list(mz_bins[1:-1]) + end)
    rt_bins = pd.interval_range(rt_start, rt_end, periods=dim[0]) 

    # create 0-matrix + helpers
    binned_matrix = np.zeros((dim[0], dim[1]))  

    # iterate over individual spectra and bin data across m/z axis, remove spectra outside of defined RT range
    for i in range(0, raw_LCMS.getNrSpectra()):

        if (raw_LCMS[i].getMSLevel() == 1) and (raw_LCMS[i].getRT() > rt_start and raw_LCMS[i].getRT() < rt_end):
            mz, intensity = raw_LCMS[i].get_peaks()

            # bin m/z values according to previously defined bins + remove values outside of range
            mz_binned = pd.cut(mz, mz_bins)
            mz_binned = mz_binned.rename_categories(list(range(0,dim[1])))
            nan_idx = np.argwhere(mz_binned.codes == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)
            grouped_intensity = [np.max([c for _, c in g]) for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])]
            grouped_bins = np.unique(mz_binned)

            # bin RT values according to previously defined bins
            rt_binned = pd.cut([raw_LCMS[i].getRT()], rt_bins)
            rt_binned = rt_binned.rename_categories(list(range(0,dim[0])))
            
            # add binned intensities to data matrix
            if np.sum(binned_matrix[rt_binned[0], :]) != 0:
                curr_val = binned_matrix[rt_binned[0], :]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity

                binned_matrix[rt_binned[0],:] = np.max(np.stack((curr_val, new_val)), axis=0)
            else:
                binned_matrix[rt_binned[0], grouped_bins] = grouped_intensity


    if save_path is not None and os.path.exists(save_path):
        if filename is not None:
            np.save(save_path + '/' + str(filename) + '.npy', binned_matrix)
        else: 
            #np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy', binned_matrix) # for mzML
            np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy', binned_matrix) # for mzData.xml

    return binned_matrix