#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Generic classifier with multiple models
    Models -> (VGG16, Xception, DenseNet201, InceptionResNetV2, InceptionV3, MobileNetV2)
    Optimizers -> "adam","sgd","rmsprop", "adagrad", "adadelta","adamax", "nadam"

    Name: main.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
            Hemerson Pistori (pistori@ucdb.br)

"""

import time
import os
import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import cv2
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import efficientnet.tfkeras as efn

from tf_explain.core.grad_cam import GradCAM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings

# =========================================================
# Constants and hyperparameters
# =========================================================

START_TIME = time.time()
IMG_WIDTH, IMG_HEIGHT = 224, 224
TRAIN_DATA_DIR = "../data/train"
TEST_DATA_DIR = "../data/test"
RESULTS_DIR = '../results/'
BATCH_SIZE = 64
EPOCHS = 1000  # Usei 1000 para as 6 classes de Eucalipto
PATIENCE_PERC = 10  # Patience related to number of epochs
PATIENCE = PATIENCE_PERC * EPOCHS / 100

LEARNING_RATE = 0.01  # Antes era 0.0001, mas muita gente usa 0.01
LEARNING_RATE_decay = 0.001
MOMENTUM = 0.9
SIZE_FC = 512
DROPOUT_FC = 0.5
MONITOR_VAR = 'val_loss' 
USE_CHECKPOINT = True 

USE_DA = True  # Change to True if want data augmentation with the parameters bellow
DA_HFLIP = True
DA_FILL_MODE = "nearest"
DA_ZOOMR = 10
DA_WSHIFTR = 0.5
DA_HSHIFTR = 0.5
DA_ROTATION = 90

VALIDATION_SPLIT = 0.2

#DA_ZOOMR = 0
#DA_WSHIFTR = 0.5
#DA_HSHIFTR = 0.5
#DA_ROTATION = 90

CLASS_NAMES = sorted(os.listdir(TRAIN_DATA_DIR))
TOTAL_CLASSES = len(CLASS_NAMES)

print("Number of Classes = %s" % TOTAL_CLASSES)
print("Classes: %s" % CLASS_NAMES)

#early_stopping_monitor = EarlyStopping(monitor=MONITOR_VAR, patience=PATIENCE, mode='auto')
early_stopping_monitor = EarlyStopping(monitor=MONITOR_VAR, patience=PATIENCE, mode='auto')

# =========================================================
# End of constants and hyperparameters
# =========================================================

def get_args():
    """Read the arguments of the program."""
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-a", "--architecture", required=True,
                           help="Select architecture(Xception, VGG16, DenseNet201, InceptionResNetV2" +
                                ", InceptionV3, MobileNetV2,NASNetLarge,EfficientNet,NASNetMobile)",
                           default=None, type=str)

    arg_parse.add_argument("-f", "--fineTuningRate", required=False,
                           help="-1: No transfer learning, 100: Transfer learning with fine tunning", default=100, type=int)

    arg_parse.add_argument("-r", "--run", required=False,
                           help="run", default=1, type=int)


    arg_parse.add_argument("-o", "--optimizer", required=False,
                           help="Optimizer: adam, sgd, rmsprop, adagrad, adadelta, adamax or nadam",
                           default="sgd", type=str)

    return vars(arg_parse.parse_args())


def get_optimizer():
    """
        Define optimizer for the neural net.
        """
    args = get_args()  # read args
    OPT_TYPE = args["optimizer"]

    if OPT_TYPE == 'adam':
        OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_decay)
    elif OPT_TYPE == 'sgd':
        OPTIMIZER = optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=LEARNING_RATE_decay, nesterov=False)
    elif OPT_TYPE == 'rmsprop':
        OPTIMIZER = optimizers.RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-08, decay=LEARNING_RATE_decay)
    elif OPT_TYPE == 'adagrad':
        OPTIMIZER = optimizers.Adagrad(lr=LEARNING_RATE, epsilon=1e-08, decay=LEARNING_RATE_decay)
    elif OPT_TYPE == 'adadelta':
        OPTIMIZER = optimizers.Adadelta(lr=LEARNING_RATE, rho=0.95, epsilon=1e-08, decay=LEARNING_RATE_decay)
    elif OPT_TYPE == 'adamax':
        OPTIMIZER = optimizers.Adamax(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                      decay=LEARNING_RATE_decay)
    elif OPT_TYPE == 'nadam':
        OPTIMIZER = optimizers.Nadam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    else:
        raise ValueError('"OPT_TYPE" %s is not valid and needs to be one of: "adam","sgd", \
                             "rmsprop","adagrad","adadelta","adamax","nadam".' % (OPT_TYPE))
    return OPTIMIZER


def plot_confusion_matrix(confusion_matrix_to_print, classes,
                          title='Confusion matrix'):
    """
        This function prints applications and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(confusion_matrix_to_print, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix_to_print.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_to_print.shape[0]),
                                  range(confusion_matrix_to_print.shape[1])):
        plt.text(j, i, format(confusion_matrix_to_print[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_to_print[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted')
    plt.xlabel('True')


def make_confusion_matrix_and_plot(test_generator, file_name, model_final):
    """Predict and plot confusion matrix"""

    test_features = model_final.predict_generator(test_generator,
                                                        test_generator.samples,
                                                        verbose=1)

    plt.figure()

    print("--------------------------------")
    print("Creating confusion matrix ....")

    cm = confusion_matrix(np.argmax(test_features, axis=1),
                          test_generator.classes)

    print("Total time after generate confusion matrix: %s" %
          (time.time() - START_TIME))

    print(cm)

    # Convert to pandas and save MATRIX as csv
    cm_df = pd.DataFrame(cm)
    matrix_csv_file = RESULTS_DIR + file_name + '_MATRIX.csv'
    with open(matrix_csv_file, mode='w') as f:
        cm_df.to_csv(f)

    plot_confusion_matrix(cm,
                          classes=CLASS_NAMES,
                          title='Confusion matrix - ' + file_name)

    plt.savefig(RESULTS_DIR + file_name + '_MATRIX.png')

    test_classes = np.argmax(test_features, axis=1)
    NAMES = np.array(CLASS_NAMES)
    test_class_names = NAMES[test_classes]

    for i, val in enumerate(test_generator.filenames):
        print(str(val) + " = " + test_class_names[i] + " " + str(test_features[i]))

    print("Total time after generate confusion matrix: %s" %
          (time.time() - START_TIME))

    print(classification_report(test_generator.classes, test_classes, target_names=CLASS_NAMES))

    precision,recall,fscore,support=score(test_generator.classes, test_classes,average='macro')
    
    # Aqui vou pegar as métricas que nos interessam e colocar no final do arquivo .csv que está
    # acumulando os resultados
    args = get_args()  # read args

    linha= str(args["run"]) + ',' + \
      args["architecture"] + ',' + \
      args["optimizer"] + ',' + \
      str(round(precision,4))+','+\
      str(round(recall,4))+','+\
      str(round(fscore,4))
      
    print(linha)
      
    f = open('../results_dl/resultados.csv','a')
    f.write(linha+'\n')
    f.close()

    


def make_training_graphics(history, file_name):
    # Convert to pandas and save HISTORY as csv
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = RESULTS_DIR + file_name + '_HISTORY.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("Stopped Epoch = %s" % early_stopping_monitor.stopped_epoch)
    used_epochs = early_stopping_monitor.stopped_epoch
    if early_stopping_monitor.stopped_epoch == 0:
        used_epochs = EPOCHS
    else:
        used_epochs = early_stopping_monitor.stopped_epoch + 1

    epochs_range = range(used_epochs)
    print(epochs_range)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(RESULTS_DIR + file_name + '_HISTORY.png')


def find_target_layer(model_final):
    for layer in reversed(model_final.layers):
        if len(layer.output_shape) == 4:
            return layer.name
		

def gradcam_samples(test_generator, file_name, model_final):
    # Create some gradcam images for furthers inspections


    explainer = GradCAM()

    test_features = model_final.predict_generator(test_generator,
                                                        test_generator.samples,
                                                        verbose=1)

    target_layer=find_target_layer(model_final)

    test_classes = np.argmax(test_features, axis=1)
    NAMES = np.array(CLASS_NAMES)
    test_class_names = NAMES[test_classes]
    output_dir="../results/gradcam_"+file_name+"_"+target_layer

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, val in enumerate(test_generator.filenames):
        img_file = TEST_DATA_DIR + "/" + str(val)
        #print(img_file + " = " + test_class_names[i] + " " + str(test_features[i]))
        img = tf.keras.preprocessing.image.load_img(img_file, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = tf.keras.preprocessing.image.img_to_array(img)
        data = ([img], None)
        
        
        grid = explainer.explain(data, model_final, class_index=1, layer_name=target_layer)
      
        
        output_file = "gradcam_"+str(val)+"_classified_as_"+test_class_names[i]+".png"
        output_file = output_file.replace("/", "_")
        explainer.save(grid, output_dir, output_file)



def main():
    """The main function"""

    model = None

    args = get_args()  # read args

    if args["fineTuningRate"] != -1:
        print('Vai carregar os pesos da imagenet')
        if args["architecture"] == "Xception":
            model = applications.Xception(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG16":
            model = applications.VGG16(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "DenseNet201":
            model = applications.DenseNet201(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionResNetV2":
            model = applications.InceptionResNetV2(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionV3":
            model = applications.InceptionV3(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "MobileNetV2":
            model = applications.MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT,3))
        elif args["architecture"] == "NASNetLarge":
            model = applications.NASNetLarge(
                weights="imagenet", include_top=False, input_shape=(331,331, 3))
        elif args["architecture"] == "NASNetMobile":
            model = applications.NASNetMobile(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT,3))
        elif args["architecture"] == "EfficientNet":
            model = efn.EfficientNetB2(
                weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT,3))


        # calculate how much layers won't be retrained according on fineTuningRate parameter
        #n_layers = len(model.layers)
        #n_layers_to_train = int(n_layers * (args["fineTuningRate"] / 100.))
        #n_layers_not_to_train = n_layers - n_layers_to_train
        #for layer in model.layers[:n_layers_not_to_train]:
        #    layer.trainable = False
        
        # NÃO ESTOU MAIS PERMITINDO UM PERCENTUAL DE AJUSTE FINO. COM fineTuningRate != -1
        # ELE ESTÁ MANTENDO TODAS AS CAMADAS TREINÁVEIS COMO TRUE (VALOR PADRÃO)

    else:  # without transfer learning
        print('NÃO vai carregar os pesos da imagenet')
        if args["architecture"] == "Xception":
            model = applications.Xception(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "VGG16":
            model = applications.VGG16(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "DenseNet201":
            model = applications.DenseNet201(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionResNetV2":
            model = applications.InceptionResNetV2(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "InceptionV3":
            model = applications.InceptionV3(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "MobileNetV2":
            model = applications.MobileNetV2(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        elif args["architecture"] == "NASNetLarge":
            model = applications.NASNetLarge(
                weights=None, include_top=False, input_shape=(331,331, 3))
        elif args["architecture"] == "EfficientNet":
            model = efn.EfficientNetB2(
                weights=None, include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
        for layer in model.layers:
            layer.trainable = True

    #print("Number of layers (without the Fully Connected) = %s" % len(model.layers))
    
    if args["fineTuningRate"] != -1:
        print("Running with ImageNet Weights and Fine Tuning All Layers at Once")
    else:
        print("Running Without Imagenet Weights ... Training from Scratch all Layers")

    # Adding custom Layers
    new_custom_layers = model.output
    new_custom_layers = Flatten()(new_custom_layers)
    new_custom_layers = Dense(SIZE_FC, activation="relu")(new_custom_layers)
    new_custom_layers = Dropout(DROPOUT_FC)(new_custom_layers)
    new_custom_layers = Dense(SIZE_FC, activation="relu")(new_custom_layers)
    predictions = Dense(TOTAL_CLASSES, activation="softmax")(new_custom_layers)

    # creating the final model
    model_final = Model(inputs=model.input, outputs=predictions)

    # compile the model
    model_final.compile(loss="categorical_crossentropy",
                        optimizer=get_optimizer(),
                        metrics=["accuracy"])

    # Initiate the train and test generators with data Augumentation
    if USE_DA:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=DA_HFLIP,
            fill_mode=DA_FILL_MODE,
            zoom_range=DA_ZOOMR,
            width_shift_range=DA_WSHIFTR,
            height_shift_range=DA_HSHIFTR,
            rotation_range=DA_ROTATION,
            validation_split=VALIDATION_SPLIT)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=VALIDATION_SPLIT)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical",
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR, 
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical',
        subset='validation') 

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode="categorical")
                 

    # select .h5 filename
    if args["fineTuningRate"] == 100:
        file_name = args["architecture"] + \
                    '_transfer_learning'
    elif args["fineTuningRate"] == -1:
        file_name = args["architecture"] + \
                    '_without_transfer_learning'
    else:
        file_name = args["architecture"] + \
                    '_fine_tunning_' + str(args["fineTuningRate"])

    file_name = file_name + "_" + args["optimizer"]

    print("\n==== HIPERPARÂMETROS ============================================\n")
    print("Running " + file_name)
    print("Batch Size = %s " % BATCH_SIZE)
    print("Number of Epochs = %s " % EPOCHS)
    print("Patience Percentual over Epochs = %s " % PATIENCE_PERC)
    print("Learning Rate = %s " % LEARNING_RATE)
    print("Learning Rate Decay = %s " % LEARNING_RATE_decay)
    print("Validation Split = %s " % VALIDATION_SPLIT)

    print("Optimization Type = " + args["optimizer"])
    print("Number of neurons fully connected layers = %s " % SIZE_FC)
    print("Dropout percentual of fully connected layers = %s " % DROPOUT_FC)
    print("Variável monitorada = %s " % MONITOR_VAR)
    print("USE CHECKPOINT = " + str(USE_CHECKPOINT))
    

    print("USE DATA AUGMENTATION = " + str(USE_DA))

    print("DATA AUGMENTATION PARAMETERS (IF USE DA IS TRUE):")
    print("Horizontal Flip = " + str(DA_HFLIP))
    print("Fill Mode = " + DA_FILL_MODE)
    print("Zoom Range = %s" % DA_ZOOMR)
    print("Width Shift Range = %s" % DA_WSHIFTR)
    print("Height Shift Range = %s" % DA_HSHIFTR)
    print("Rotation range = %s" % DA_ROTATION)

    model_final.summary()
    
    # Status of layers regarding training
    print("List of layers with trainable status")
    for layer in model.layers:
        print(str(layer.name) + " = " + str(layer.trainable))

       
    checkpoint_file_name = "../models_checkpoints/" + file_name + ".h5"
    # Save the model according to the conditions
    checkpoint = ModelCheckpoint(checkpoint_file_name, monitor=MONITOR_VAR,
                                 verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)

    # Train the model
    history = model_final.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping_monitor, checkpoint],
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // BATCH_SIZE)

    print("Total time to train: %s" % (time.time() - START_TIME))

    if USE_CHECKPOINT: model_final = load_model(checkpoint_file_name)

    make_training_graphics(history, file_name)

    # Create and save a confusion matrix using the test set
    test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        TEST_DATA_DIR,
        batch_size=1,
        shuffle=False,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical")

    make_confusion_matrix_and_plot(
        test_generator, file_name, model_final)

    gradcam_samples(test_generator, file_name, model_final)


if __name__ == '__main__':
    main()
