import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_data, utils
from utils import load_model as loader
from tensorflow.keras.models import load_model

def args_setting():
    """ Parse arguments from command line.

    Return: \\
    arguments
    """

    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath', default='./meta_test.xlsx')
    parser.add_argument('-data', type=list, dest='raw_folderpaths', default=[
            "./Other/data/acidosis_adaptive",
            "./Other/data/drug_adaptive",
            "./Other/data/hanging_adaptive",
            "./Other/data/ihd_adaptive",
            "./Other/data/pneumonia_adaptive"])
    parser.add_argument('-test_samples', type=str, dest='sample_path', default=None)
    parser.add_argument('-models', type=str, dest='model_dir', default='./CNN/results/final_ensemble/Run 5')
    parser.add_argument('-save', dest='save_path', default='./CNN/results/final_ensemble/Run 5')
    parser.add_argument('-mode', type=str, dest='mode', default='ensemble')
    args = parser.parse_args()

    return args


def main():

    # retrieve args
    args = args_setting()
    meta_filepath = args.meta_filepath
    raw_folderpaths = args.raw_folderpaths
    model_dir = args.model_dir
    save_path = args.save_path
    mode = args.mode

    # load training arguments
    batch_size = 4
    use_norm = True

    # load test data
    meta = pd.read_excel(meta_filepath, index_col=0)
    class_list = np.unique(meta["Group"])

    if args.sample_path is not None:
        samples_test = pd.read_csv(args.sample_path, sep=",", header=None)[0].to_list()
    else:
        samples_test = meta.index.to_list()
       
    with tf.device('/CPU:0'):
        test_dataset = tf.data.Dataset.from_generator(
            load_data.data_gen, 
            args=(raw_folderpaths, meta_filepath, samples_test, use_norm, False), 
            output_signature=(tf.TensorSpec(shape=(856, 1024), dtype=tf.float64), tf.TensorSpec(shape=(len(class_list),), dtype=tf.int32)))
        test_dataset = test_dataset.batch(batch_size, drop_remainder=False).prefetch(1)

    # load model
    if mode == 'single':
        model_path = os.path.join(model_dir, 'model.h5')
        if not os.path.exists(model_path):
            raise ValueError("model_path not exist")
        
        model = loader.SingleModel(model_path)
    elif mode == 'ensemble':

        model_list = []
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name, 'model.h5')
            if os.path.exists(model_path):
                model_list.append(model_path)
        
        model = loader.EnsembleModel(model_list)
    else:
        raise ValueError("mode must be ensemble or single")

    # predict samples and evaluate performance
    utils.eval(model, test_dataset, np.concatenate([y for x, y in test_dataset], axis=0), "CNN_test", save_path, False)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
