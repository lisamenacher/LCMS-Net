import pandas as pd
import numpy as np
import pyopenms as oms
import os
import argparse
import os
import time

from itertools import groupby


def args_setting():
    """ Parse arguments from command line.

    Return: \\
    arguments
    """

    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath', default='./Example Data/meta.xlsx')
    parser.add_argument('-data', type=str, dest='raw_folderpaths', default="./Raw Example")
    parser.add_argument('-save', dest='save_path', default='./Binned Data')
    parser.add_argument('-mode', type=str, dest='mode', default='default')

    args = parser.parse_args()

    return args


def main():
    # retrieve args
    args = args_setting()
    meta_filepath = args.meta_filepath
    folder = args.raw_folderpaths
    save_path = args.save_path
    mode = args.mode

    # load meta data (class labels, PMI etc.)
    data = pd.read_excel(meta_filepath, index_col=0)

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
                    if mode == "adaptive":    
                        intensity_matrix = __adaptive_binning(exp, save_path=save_path, filetype="mzML")
                    else:
                        intensity_matrix = __default_binning(exp, save_path=save_path, filetype="mzML")
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
                if mode == "adaptive":    
                    intensity_matrix = __adaptive_binning(exp, save_path=save_path, filetype="mzdata")
                else:
                    intensity_matrix = __default_binning(exp, save_path=save_path, filetype="mzdata")
            except:
                print(f"ERROR: File {file} could not be loaded!")




    
def __adaptive_binning(raw_LCMS, dim=(856, 1024), rt_start=45, rt_end=660, mz_start=50, mz_end=1000, save_path=None, filename=None, filetype="mzML"):

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
            if filetype == "mzML":
                np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy', binned_matrix) # for mzML 
            elif filetype == "mzdata":
                np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy', binned_matrix) # for mzData.xml

    return binned_matrix



def __default_binning(raw_LCMS, dim=(1024, 1024), rt_start=45, rt_end=660, mz_start=49.9, mz_end=1000, save_path=None, filename=None, filetype="mzML"):
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
            if filetype == "mzML":
                np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy', binned_matrix) # for mzML 
            elif filetype == "mzdata":
                np.save(save_path + '/' + raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy', binned_matrix) # for mzData.xml

    return binned_matrix   

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
