import platform
import os
import time
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import create_model
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from utils import load_data, utils


if platform.system().lower() == 'linux':
    print("[INFO] run in linux platform!")


def args_setting():
    """ Parse arguments from command line and set default values.

    Return: \\ 
    arguments
    """

    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath', default='./meta_train.xlsx')
    parser.add_argument('-data', type=list, dest='raw_folderpaths', default=[
            "./Other/data/acidosis_adaptive",
            "./Other/data/drug_adaptive",
            "./Other/data/hanging_adaptive",
            "./Other/data/ihd_adaptive",
            "./Other/data/pneumonia_adaptive"])
    parser.add_argument('-handleImbalance', type=str, dest='handle_imbalance', default=None)
    parser.add_argument('-norm', type=bool, dest='use_norm', default=False)
    parser.add_argument('-aug', type=bool, dest='use_aug', default=False)
    parser.add_argument('-lr', type=float, dest='lr', default=0.0001)
    parser.add_argument('-batch', type=int, dest='batch_size', default=4)
    parser.add_argument('-epoch', type=int, dest='epoch', default=200)
    parser.add_argument('-save', type=str, dest='save_dir', default='./CNN/results/')
    args = parser.parse_args()
    
    return args


def get_run_log(history):
    """ Create a dataframe with the training history of a model.

    Parameter: \\
    history:           Keras model history

    Return: \\
    pd.Dataframe
    """

    run_log = pd.DataFrame()
    run_log['epoch'] = list(range(1, 1 + len(history.history['loss'])))
    run_log['loss'] = history.history['loss']
    run_log['accuracy'] = history.history['accuracy']
    run_log['f1_score'] = history.history['f1_score']
    run_log['AUC-ROC'] = history.history['AUC-ROC']
    run_log['AUC-PR'] = history.history['AUC-PR']
    run_log['val_loss'] = history.history['val_loss']
    run_log['val_accuracy'] = history.history['val_accuracy']
    run_log['val_f1_score'] = history.history['val_f1_score']
    run_log['val_AUC-ROC'] = history.history['val_AUC-ROC']
    run_log['val_AUC-PR'] = history.history['val_AUC-PR']

    return run_log


def main():
    # retrieve arguments
    args = args_setting()
    args_df = utils.create_args_df(args) 

    # data paths + shape
    meta_filepath = args.meta_filepath
    raw_folderpaths = args.raw_folderpaths

    input_shape = (856, 1024)
    
    # training setup
    use_norm = args.use_norm
    use_aug = args.use_aug
    handle_imbalance = args.handle_imbalance
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batch_size

    # create savedir if it does not exist already
    save_dir = args.save_dir 
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load data und check how to handle class imbalance
    meta = pd.read_excel(meta_filepath, index_col=0)
    num_classes = len(np.unique(meta["Group"]))

    for i in range(11):

        samples_train, samples_val = utils.split_samples(meta.index, meta["Group"], split_size=0.1)
        random.shuffle(samples_train)
        
        class_weights = None
        if handle_imbalance == "ros":
            # perform oversampling
            ros = RandomOverSampler()
            samples_train_resampled, y_resampled = ros.fit_resample(np.array(samples_train).reshape(-1, 1), meta.loc[samples_train]["Group"].to_list())
            samples_train = samples_train_resampled.flatten()
            random.shuffle(samples_train)
        elif handle_imbalance == "class_weights":
            # use class weights
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(meta["Group"]), y=meta["Group"].tolist())
            class_weights = dict(enumerate(weights))

        with tf.device('/CPU:0'):
            val_dataset = tf.data.Dataset.from_generator(
                load_data.data_gen, 
                args=(raw_folderpaths, meta_filepath, samples_val, use_norm, False), 
                output_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float64), tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)))
            val_dataset = val_dataset.batch(batch_size, drop_remainder=False).prefetch(1).cache()

            train_dataset = tf.data.Dataset.from_generator(
                load_data.data_gen, 
                args=(raw_folderpaths, meta_filepath, samples_train, use_norm, use_aug), 
                output_signature=(tf.TensorSpec(shape=input_shape, dtype=tf.float64), tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=False).prefetch(3).cache( "../../vault/lisme75_vr-murderai/3. Projects - IDA/Cause of Death - Lisa/data/train.cache")

        # create and compile model
        model = create_model.create_model(input_shape=input_shape, classes=len(np.unique(meta["Group"])))
        model = create_model.compile_model(model, lr=lr)

        # fit model
        early_stopping = EarlyStopping(monitor='val_f1_score', patience=2, start_from_epoch=1, restore_best_weights=True, mode='max')
        lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=0.0000001)
        history = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset, callbacks=[lr_schedule, early_stopping], verbose=1, class_weight=class_weights) 
        run_log = get_run_log(history)

        # store results
        print('[INFO] run log:')
        print(run_log)

        # save model if save_dir is specified    
        model_id = str(int(time.time()))+str(np.random.randint(0,1000))
        if save_dir is not None:
            # create save dir
            model_savedir = os.path.join(save_dir, f"CNN_{model_id}")
            if not os.path.exists(model_savedir):
                os.makedirs(model_savedir)

            # save training arguments
            args_path = os.path.join(model_savedir, 'args.txt')
            args_df.to_csv(args_path, index=False, sep='\t')

            # save training log
            log_path = os.path.join(model_savedir, 'log.txt')
            run_log.to_csv(log_path, index=False, sep='\t')

            plt.plot(run_log["epoch"], run_log["loss"], label='Train')
            plt.plot(run_log["epoch"], run_log["val_loss"], label='Valid')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc="upper left")
            plt.savefig(model_savedir + "/loss.png")
            plt.close()

            plt.plot(run_log["epoch"], run_log["f1_score"], label='Train')
            plt.plot(run_log["epoch"], run_log["val_f1_score"], label='Valid')
            plt.xlabel("Epoch")
            plt.ylabel("F1-Score")
            plt.legend(loc="upper left")
            plt.savefig(model_savedir + "/f1_score.png")
            plt.close()

            # utils.eval(model, train_dataset, np.concatenate([y for x, y in train_dataset], axis=0), "CNN_train", model_savedir)
            utils.eval(model, val_dataset, np.concatenate([y for x, y in val_dataset], axis=0), "CNN_val", model_savedir)

            # save model
            model_path = os.path.join(model_savedir, 'model.h5')
            model.save(model_path)

            # save the used train/test split
            utils.save_split_data(samples_train, samples_val, model_path.replace('.h5', ''))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("[INFO] RUNNING in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("[INFO] END in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
