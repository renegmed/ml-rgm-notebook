
import tensorflow as tf
import argparse
import os
import numpy as np
import json

#import matplotlib.pyplot as plt 
#import subprocess

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def model(x_train, y_train, x_test, y_test):
    """Generate a simple model"""
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1024, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])

#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.fit(x_train, y_train)
#     model.evaluate(x_test, y_test)

#     return model

    K = len(set(y_train))
    # Build the model using the functional API
    i = Input(shape=x_train[0].shape)  # tensorflow.keras.layers.Input
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(K, activation='softmax')(x)

    model = Model(i, x)

    model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
    
    r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    return model


def _load_data(base_dir):
    # Load in the data
#     fashion_mnist = tf.keras.datasets.fashion_mnist
#     (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.load(os.path.join(base_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'y_train.npy'))
    x_test = np.load(os.path.join(base_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'y_test.npy'))
    
    x_train, x_test = x_train / 255.0, x_test/ 255.0

    # the data is only 2D!
    # convolution expects height x width x color
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return x_train, y_train, x_test, y_test


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()

    train_data, train_labels, eval_data, eval_labels = _load_data(args.train)
#     eval_data, eval_labels = _load_testing_data(args.train)

    mnist_classifier = model(train_data, train_labels, eval_data, eval_labels)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        mnist_classifier.save(os.path.join(args.sm_model_dir, '000000002'), 'mnist_model.h5')
