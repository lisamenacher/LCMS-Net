import pandas as pd
import numpy as np
import os

from tensorflow.keras.utils import to_categorical

  
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


