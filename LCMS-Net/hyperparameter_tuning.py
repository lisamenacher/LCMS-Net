import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import kerastuner as kt
import json 
import random

from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import load_data, utils, create_model

def args_setting():
    """ Parse arguments from command line and set default values.

    Return: \\
    arguments
    """

    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath', default='./LCMS-Net/meta.xlsx')
    parser.add_argument('-data', type=list, dest='raw_folderpaths', default=["./LCMS-Net"])
    parser.add_argument('-batch', type=int, dest='batch_size', default=4)
    parser.add_argument('-epoch', type=int, dest='epoch', default=200)
    parser.add_argument('-save', type=str, dest='save_dir', default='./results')
    args = parser.parse_args()
    
    return args


def main():
    # retrieve arguments
    args = args_setting()
    args_df = utils.create_args_df(args) 
    meta_filepath = args.meta_filepath
    raw_folderpaths = args.raw_folderpaths
    epochs = args.epoch
    batch_size = args.batch_size

    # load data
    meta = pd.read_excel(meta_filepath, index_col=0)
    num_classes = len(np.unique(meta["Group"]))
    samples_train, samples_val = utils.split_samples(meta.index, meta["Group"], split_size=0.2, seed=42)

    ros = RandomOverSampler()
    samples_train_resampled, y_resampled = ros.fit_resample(np.array(samples_train).reshape(-1, 1), meta.loc[samples_train]["Group"].to_list())
    samples_train = samples_train_resampled.flatten()
    random.shuffle(samples_train)

    with tf.device('/CPU:0'):
        val_dataset = tf.data.Dataset.from_generator(
            load_data.data_gen, 
            args=(raw_folderpaths, meta_filepath, samples_val, False, False), 
            output_signature=(tf.TensorSpec(shape=(856, 1024), dtype=tf.float64), tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)))
        val_dataset = val_dataset.batch(batch_size, drop_remainder=False).prefetch(1).cache()

        train_dataset = tf.data.Dataset.from_generator(
            load_data.data_gen, 
            args=(raw_folderpaths, meta_filepath, samples_train, False, False), 
            output_signature=(tf.TensorSpec(shape=(856, 1024), dtype=tf.float64), tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)))
        train_dataset = train_dataset.batch(batch_size, drop_remainder=False).prefetch(3).cache()

    # load and fit tuner
    tuner = kt.BayesianOptimization(
        create_model.create_base_hypermodel,
        objective=kt.Objective('val_f1_score', direction='max'),
        max_trials=100,
        max_consecutive_failed_trials=5)
    
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=10, start_from_epoch=3)
    lr_schedule = ReduceLROnPlateau(monitor='val_f1_score', factor=0.5, patience=3, verbose=1, min_lr=0.0000001)
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[lr_schedule, early_stopping],
        class_weight=None,
        verbose=0)

    # save results
    model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

    if args.save_dir is not None:
        model_savedir = os.path.join(args.save_dir, f"hyperparam_search")
        if not os.path.exists(model_savedir):
            os.makedirs(model_savedir)

            utils.eval(model, train_dataset, np.concatenate([y for x, y in train_dataset], axis=0), np.unique(meta["Group"]), "CNN_train", model_savedir)
            utils.eval(model, val_dataset, np.concatenate([y for x, y in val_dataset], axis=0), np.unique(meta["Group"]), "CNN_val", model_savedir)

        args_path = os.path.join(model_savedir, 'args.txt')
        args_df.to_csv(args_path, index=False, sep='\t')

        model_path = os.path.join(model_savedir, 'model.h5')
        model.save(model_path)

        log_path = os.path.join(model_savedir, 'log.json')
        params = best_hyperparameters.values
        json.dump(params, open(log_path, 'w' ) )
    

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("[INFO] RUNNING in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("[INFO] END in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

