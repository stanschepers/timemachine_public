import os
from datetime import datetime

import tensorflow as tf
import keras
from keras import layers


INPUT_SHAPE = (224, 224, 3)


def get_default_top():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            
        ], name="default_top"
    ) 


def get_data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            
        ], name="data_augmentation"
    )


def get_path(name):
    date = datetime.now().strftime("%d%m%y")
    now_time = datetime.now().strftime("%H%M")
    return os.path.join(name, date, now_time)


def get_callbacks(path):
    logs_dir = os.path.join("logs", path)
    model_dir = os.path.join("saved_models", path)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    return [
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        tf.keras.callbacks.ModelCheckpoint(model_dir, monitor='val_loss', save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)
    ]
