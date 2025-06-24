import sys
import os
import tensorflow as tf
import keras
import json

from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, Dropout, BatchNormalization, ReLU, Conv2D, MaxPooling1D, DepthwiseConv1D, SpatialDropout1D, MaxPooling2D, Reshape
from keras.models import Model
from keras.initializers import Ones, Zeros
from keras.losses import CategoricalCrossentropy
from keras.metrics import AUC, F1Score, Accuracy, CategoricalAccuracy 

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def create_base_hypermodel(hp):
    """ Builds model for hyperparameter tuning with keras tuner

    Parameter: \\
    hp (HyperParameters):           keras_tuner.HyperParameters class \\


    Return: \\
    keras model
    """

    # inialize input
    input_tensor = Input(shape=(856, 1024))
    x = input_tensor

    # add conv_blocks
    kernel_size = hp.Int('kernel_size', 2, 60, step=2, default=10)
    if kernel_size < 10:
        kernel_stride = hp.Int('kernel_stride', 2, kernel_size, step=1, default=2)
    else:
        kernel_stride = hp.Int('kernel_stride', 2, 10, step=1, default=2)
    pool_size = hp.Int('pool_size', 2, 30, step=1, default=2)
    if pool_size < 10:
        pool_stride = hp.Int('pool_stride', 2, pool_size, step=1, default=2)
    else:
        pool_stride = hp.Int('pool_stride', 2, 10, step=1, default=2)
    droput_rate = hp.Float('dropout', 0, 0.5, default=0.5)
    kernel_reg_l1 = hp.Float('kernel_reg_l1', 0, 0.1, default=0.005)
    kernel_reg_l2 = hp.Float('kernel_reg_l2', 0, 0.1, default=0.005)
    dense_reg_l1 = hp.Float('dense_reg_l1', 0, 0.1, default=0.005)
    dense_reg_l2 = hp.Float('dense_reg_l2', 0, 0.1, default=0.005)


    x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
    x = ReLU()(x) 
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
    x = SpatialDropout1D(droput_rate)(x)
    x = Flatten()(x)

    # add classification head
    x = Dense(2, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1_l2(dense_reg_l1, dense_reg_l2))(x)
    
    # create and compile model
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            0.0001, # will be decreased upon plateu anyways
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=hp.Float("epsilon", 1e-9, 0.01, default=1e-7), 
            weight_decay=hp.Float("weight_decay", 0, 0.25, default=1e-7), 
            amsgrad=False),
        loss=CategoricalCrossentropy(), 
        metrics=[CategoricalAccuracy(name="accuracy"), F1Score(average="macro", name="f1_score"), AUC(curve="ROC", name='AUC-ROC'), AUC(curve="PR", name='AUC-PR')])

    return model


def create_multilayer_hypermodel(hp):
    """ Builds model for hyperparameter tuning with keras tuner

    Parameter: \\
    hp (HyperParameters):           keras_tuner.HyperParameters class \\


    Return: \\
    keras model
    """

    # inialize input
    input_tensor = Input(shape=(856, 1024))
    x = input_tensor

    # add conv_blocks
    num_conv_blocks = hp.Int("num_conv_blocks", 1, 5, default=3)

    kernel_reg_l1 = hp.Float('kernel_reg_l1_1', 0, 0.1, default=0.005)
    kernel_reg_l2 = hp.Float('kernel_reg_l2_1', 0, 0.1, default=0.005)
    kernel_size = hp.Int('kernel_size_0', 2, 60, step=2, default=10)
    if kernel_size < 30:
        kernel_stride = hp.Int('kernel_stride_1', 2, kernel_size, step=1, default=2)
    else:
        kernel_stride = hp.Int('kernel_stride_1', 2, 30, step=1, default=2)

    x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
    x = ReLU()(x) 
    x = BatchNormalization()(x)

    pool_size = hp.Int('pool_size_0', 2, 10, step=1, default=2)
    if pool_size < 5:
        pool_stride = hp.Int('pool_stride_1', 1, pool_size, step=1, default=2)
    else:
        pool_stride = hp.Int('pool_stride_1', 1, 5, step=1, default=2)
    droput_rate = hp.Float('dropout_0', 0, 0.5, default=0.5)

    x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
    x = SpatialDropout1D(droput_rate)(x) 

    with hp.conditional_scope("num_conv_blocks", [2, 3, 4, 5]):
        if num_conv_blocks > 1:

            kernel_reg_l1 = hp.Float('kernel_reg_l1_2', 0, 0.1, default=0.005)
            kernel_reg_l2 = hp.Float('kernel_reg_l2_2', 0, 0.1, default=0.005)
            max_size = int(x.shape[1]/2) if int(x.shape[1]/2) < 30 else 30
            kernel_size = hp.Int('kernel_size_2', 1, max_size)
            max_size =  int(kernel_size/2) if int(kernel_size/2) > 1 else 1
            kernel_stride = hp.Int('kernel_stride_2', 1, max_size, default=3)

            x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
            x = ReLU()(x) 
            x = BatchNormalization()(x)

            max_size =  int(x.shape[1]/2) if int(x.shape[1]/2) > 1 else 1
            pool_size = hp.Int('pool_size_2', 1, max_size, step=1, default=2)
            max_size =  int(pool_size/2) if int(pool_size/2) > 1 else 1
            pool_stride = hp.Int('pool_stride_2', 1, max_size, step=1, default=2)
            droput_rate = hp.Float('dropout_2' , 0, 0.5, default=0.5)

            x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
            x = SpatialDropout1D(droput_rate)(x)

    with hp.conditional_scope("num_conv_blocks", [3, 4, 5]):
        if num_conv_blocks > 2:

            kernel_reg_l1 = hp.Float('kernel_reg_l1_3', 0, 0.1, default=0.005)
            kernel_reg_l2 = hp.Float('kernel_reg_l2_3', 0, 0.1, default=0.005)
            max_size = int(x.shape[1]/2) if int(x.shape[1]/2) < 30 else 30
            kernel_size = hp.Int('kernel_size_3', 1, max_size)
            max_size =  int(kernel_size/2) if int(kernel_size/2) > 1 else 1
            kernel_stride = hp.Int('kernel_stride_3', 1, max_size, default=3)

            x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
            x = ReLU()(x) 
            x = BatchNormalization()(x)

            max_size =  int(x.shape[1]/2) if int(x.shape[1]/2) > 1 else 1
            pool_size = hp.Int('pool_size_3', 1, max_size, step=1, default=2)
            max_size =  int(pool_size/2) if int(pool_size/2) > 1 else 1
            pool_stride = hp.Int('pool_stride_3', 1, max_size, step=1, default=2)
            droput_rate = hp.Float('dropout_3' , 0, 0.5, default=0.5)

            x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
            x = SpatialDropout1D(droput_rate)(x)          
    
    with hp.conditional_scope("num_conv_blocks", [4, 5]):
        if num_conv_blocks > 3:

            kernel_reg_l1 = hp.Float('kernel_reg_l1_4', 0, 0.1, default=0.005)
            kernel_reg_l2 = hp.Float('kernel_reg_l2_4', 0, 0.1, default=0.005)
            max_size = int(x.shape[1]/2) if int(x.shape[1]/2) < 30 else 30
            kernel_size = hp.Int('kernel_size_4', 1, max_size)
            max_size =  int(kernel_size/2) if int(kernel_size/2) > 1 else 1
            kernel_stride = hp.Int('kernel_stride_4', 1, max_size, default=3)

            x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
            x = ReLU()(x) 
            x = BatchNormalization()(x)

            max_size =  int(x.shape[1]/2) if int(x.shape[1]/2) > 1 else 1
            pool_size = hp.Int('pool_size_4', 1, max_size, step=1, default=2)
            max_size =  int(pool_size/2) if int(pool_size/2) > 1 else 1
            pool_stride = hp.Int('pool_stride_4', 1, max_size, step=1, default=2)
            droput_rate = hp.Float('dropout_4' , 0, 0.5, default=0.5)

            x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
            x = SpatialDropout1D(droput_rate)(x)   
    
    with hp.conditional_scope("num_conv_blocks", [5]):
        if num_conv_blocks == 5:

            kernel_reg_l1 = hp.Float('kernel_reg_l1_5', 0, 0.1, default=0.005)
            kernel_reg_l2 = hp.Float('kernel_reg_l2_5', 0, 0.1, default=0.005)
            max_size = int(x.shape[1]/2) if int(x.shape[1]/2) < 30 else 30
            kernel_size = hp.Int('kernel_size_5', 1, max_size)
            max_size =  int(kernel_size/2) if int(kernel_size/2) > 1 else 1
            kernel_stride = hp.Int('kernel_stride_5', 1, max_size, default=3)

            x = DepthwiseConv1D(kernel_size=kernel_size, depth_multiplier=1, strides=kernel_stride, depthwise_regularizer=tf.keras.regularizers.l1_l2(kernel_reg_l1, kernel_reg_l2), depthwise_initializer=keras.initializers.he_normal())(x)
            x = ReLU()(x) 
            x = BatchNormalization()(x)

            max_size =  int(x.shape[1]/2) if int(x.shape[1]/2) > 1 else 1
            pool_size = hp.Int('pool_size_5', 1, max_size, step=1, default=2)
            max_size =  int(pool_size/2) if int(pool_size/2) > 1 else 1
            pool_stride = hp.Int('pool_stride_5', 1, max_size, step=1, default=2)
            droput_rate = hp.Float('dropout_5' , 0, 0.5, default=0.5)

            x = MaxPooling1D(pool_size=pool_size, strides=pool_stride)(x)
            x = SpatialDropout1D(droput_rate)(x)   

    # add classification head
    x = Flatten()(x)
    num_layers = hp.Int("num_dense_layers", 0, 3, default=1)
    with hp.conditional_scope("num_dense_layers", [1, 2, 3]):
        if num_layers != 0:
            num_hidden_units = hp.Int('num_hidden_units_1', 1,64)
            droput_rate = hp.Float('dense_dropout_1', 0, 0.5, step=0.1, default=0.5)
            x = Dense(num_hidden_units, activation='relu')(x)
            x = Dropout(droput_rate)(x)
    
    with hp.conditional_scope("num_dense_layers", [2, 3]):
        if num_layers > 1:
            num_hidden_units_1 = hp.Int('num_hidden_units_2', 1,1024)
            droput_rate_1 = hp.Float('dense_dropout_2', 0, 0.5, step=0.1, default=0.5)
            x = Dense(num_hidden_units_1, activation='relu')(x)
            x = Dropout(droput_rate_1)(x)

    with hp.conditional_scope("num_dense_layers", [3]):
        if num_layers == 3:
            num_hidden_units_2 = hp.Int('num_hidden_units_3', 1,1024)
            droput_rate_2 = hp.Float('dense_dropout_3', 0, 0.5, step=0.1, default=0.5)
            x = Dense(num_hidden_units_2, activation='relu')(x)
            x = Dropout(droput_rate_2)(x)

    dense_reg_l1 = hp.Float('output_reg_l1', 0, 0.1, default=0.005)
    dense_reg_l2 = hp.Float('output_reg_l2', 0, 0.1, default=0.005)
    x = Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1_l2(dense_reg_l1, dense_reg_l2))(x)
    
    # create and compile model
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False, epsilon=0.005, weight_decay=0.11), 
        loss=CategoricalCrossentropy(), 
        metrics=['accuracy', F1Score(average="macro"), AUC(curve="ROC", name='AUC-ROC'), AUC(curve="PR", name='AUC-PR')])

    return model


def load_tuned_model(param_json, input_shape=(856, 1024), classes=5):
    """ Retrieves best hyperparameters from json and builds the model accordingly

    Parameter: \\
    paran_json (str):           Path of hyperparameter json file \\
    input_shape (tuple):        Shape of processed LC-MS data \\
    classes (int):              Number of class labels \\

    Return: \\
    keras model
    """

    # read json with stored hyperparams
    with open(param_json) as f:
        hyperparameters = json.load(f)

    # inialize input
    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # add conv_blocks
    for i in range(hyperparameters["num_conv_blocks"]):
        num_filters = hyperparameters['num_filters_' + str(i)]
        kernel_size = hyperparameters['kernel_sizes_' + str(i)]
        droput_rate = hyperparameters['cnn_dropout_' + str(i)]

        x = DepthwiseConv1D(num_filters, kernel_size, padding='same')(x)
        x = MaxPooling1D()(x)
        x = BatchNormalization()(x)
        x = ReLU()(x) 
        x = Dropout(droput_rate)(x)

    # add classification head
    x = Flatten()(x)

    for i in range(hyperparameters["num_dense_layers"]):
        num_hidden_units = hyperparameters['num_hidden_units' + str(i)]
        droput_rate = hyperparameters['dense_dropout_' + str(i)] 

        x = Dense(num_hidden_units, activation='relu')(x)
        x = Dropout(droput_rate)(x)

    x = Dense(classes, activation='softmax')(x)
    
    # create and compile model
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hyperparameters["learning_rate"]),
        loss='categorical_crossentropy', 
        metrics=[Accuracy(), F1Score(average="weighted", name="f1_score"), AUC(curve="ROC", name='AUC-ROC'), AUC(curve="PR", name='AUC-PR')])

    return model


def create_model(input_shape, classes=5):
    """ Create 1D-CNN for CoD prediction.

    Parameter: \\
    input_shape (tuple):        Shape of processed LC-MS data \\
    classes (int):              Number of class labels \\
    
    Return: \\
    keras model
    """

    # inialize model with specified parameters
    input_tensor = Input(shape=(input_shape[0], input_shape[1]), name="input_lcms")
    x = DepthwiseConv1D(kernel_size=42, depth_multiplier=1, strides=5, depthwise_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.05), depthwise_initializer=keras.initializers.he_normal(), padding="same")(input_tensor)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=6, strides=2, padding="same")(x)
    x = SpatialDropout1D(0.1)(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1_l2(0.002, 0.01))(x) 
    model = Model(inputs=input_tensor, outputs=x)

    return model


def compile_model(model, lr=1e-3):
    """ Compile model with specified parameters. 

    Parameter: \\
    model (keras model):        Initalized Keras Classification Model \\
    lr (float):                 Learning Rate \\

    Return: \\
    compiled model
    """
    loss_func = CategoricalCrossentropy()
    metrics=['accuracy', F1Score(average="macro"), AUC(curve="ROC", name='AUC-ROC'), AUC(curve="PR", name='AUC-PR')]
    model.compile(optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False, epsilon=0.005, weight_decay=0.11), loss=loss_func, metrics=metrics)
    
    return model 


