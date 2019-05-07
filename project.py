import MNIST_Loader as ml
# Package for third-party implementation of some of the neural network nitty-gritty
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


# Code for the final project, which uses a convolutional neural network
# to identify handwritten letters using EMNIST as the data set

# NOTE: This code was started with the help of this site:
# https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/
# We, as Group 13, have made significant edits, but credit to Sentdex for the great tutorial.

# Label Key
# label = (entry #, key)
# 0 - 9 = 0 - 9
# A - Z = 10 - 35
# a - z = 36 - 47
# NOTE: not all lowercase letters have a number different
# from their uppercase counterpart


# Authors:
#   Chase Abram
#   Christopher Anthony
#   Austin Russell

class EmnistModelGenerator:
    """
    A class to generate a model to recognize handwritten characters
    based on the EMNIST dataset.

    Attributes:
        model (CNN): A convultional neural network built with Keras using the given parameters
        training_images (Numpy array): The training data from EMNIST's balanced dataset's first 80% of images
        training_labels (Numpy array): The training labels from EMNIST's balanced dataset's first 80% of labels
        validation_images (Numpy array): The validation data from EMNIST's balanced dataset's last 20% of images
        validation_labels (Numpy array): The validation labels from EMNIST's balanced dataset's last 20% of labels
        test_images (Numpy array): The testing data from EMNIST's balanced dataset
        test_labels (Numpy array): The testing labels from EMNIST's balanced dataset
        optimizer (str): Optimizer to be used by Keras
        loss (str): Loss function to be used by Keras
        metrics (list of str): Metrics to be used by Keras to judge the performance of the model
        activation (function): Function used by Keras to determine whether or not a neuron 'fires'
        neurons (int): The number of neurons to be used in the hidden layers
        dense_layers (int): The number of dense layers to be used in the network
        epochs (int): The number of times the network passes through the training data
        convolutional_pooling (int): The number of convolutional, pooling, and dropout layers to add to the network
        filters (int): The filters argument to use with the convolutional layers
        kernel_size (int): the kernel size to use with the convolutional layers
        pool_size(int): the pooling size to use with the pooling layers
        input_shape (int): the shape of the input data (always the same for the EMNIST data)
    """

    CLASSIFICATIONS = 47

    def __init__(self, optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'],
                 activation=tf.nn.relu,
                 neurons=128,
                 dense_layers=2,
                 convolutional_pooling=2,
                 filters=30,
                 kernel_size=2,
                 pool_size=(2, 2),
                 epochs=10,
                 input_shape=(28, 28, 1),):
        self.model = self.initialize_model()
        self.training_images, self.training_labels = self.initialize_training()
        self.test_images, self.test_labels = self.initialize_test()
        self.validation_images, self.validation_labels = self.initialize_validation()
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.neurons = neurons
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.convolutional_pooling = convolutional_pooling
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.generate_model()

    """Generates the model based on the input to the constructor."""

    def generate_model(self):

        self.add_initial_convolutional_layer()

        # convolutional_pooling represents the number of times we add these three layers.
        for i in range(self.convolutional_pooling):
            self.add_convolutional_layer()
            self.add_pooling_layer()
            self.add_dropout_layer()

        self.flatten_layer()

        for i in range(self.dense_layers):
            self.add_dense_layer()

        self.finish_model()
        self.save_model()

    """Initializes the model."""

    def initialize_model(self):
        model = tf.keras.models.Sequential()
        return model

    """Initializes the training data."""

    def initialize_training(self):
        # training_data and testing_data are tuples containing two elements:
        # - the numpy array of data
        # - the numpy array of labels
        training_data = ml.load()[0]

        training_images = tf.keras.utils.normalize(training_data[0], axis=1)
        training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
        training_labels = training_data[1]

        # NOTE: This implicitly returns a tuple
        return training_images, training_labels

    """Initializes the validation data."""

    def initialize_validation(self):
        # training_data and testing_data are tuples containing two elements:
        # - the numpy array of data
        # - the numpy array of labels
        validation_data = ml.load()[2]

        validation_images = tf.keras.utils.normalize(validation_data[0], axis=1)
        validation_images = validation_images.reshape(validation_images.shape[0], 28, 28, 1)
        validation_labels = validation_data[1]

        # NOTE: This implicitly returns a tuple
        return validation_images, validation_labels

    """Initializes the testing data."""

    def initialize_test(self):
        # training_data and testing_data are tuples containing two elements:
        # - the numpy array of data
        # - the numpy array of labels
        test_data = ml.load()[1]

        test_images = tf.keras.utils.normalize(test_data[0], axis=1)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
        test_labels = test_data[1]

        # NOTE: This implicitly returns a tuple
        return test_images, test_labels

    """Adds a single hidden layer to the network."""

    def add_dense_layer(self):
        self.model.add(tf.keras.layers.Dense(self.neurons, self.activation))

    """Adds a single dropout layer to the network."""

    def add_dropout_layer(self):
        self.model.add(tf.keras.layers.Dropout(0.25))

    """Flattens model."""

    def flatten_layer(self):
        self.model.add(tf.keras.layers.Flatten())

    """Adds a single convolutional layer to the network."""

    def add_initial_convolutional_layer(self):
        self.model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                              activation=self.activation,
                                              input_shape=self.input_shape))

    def add_convolutional_layer(self):
        self.model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=self.activation))

    """Adds a single pooling layer to the network."""

    def add_pooling_layer(self):
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=self.pool_size))

    """
    Adds the final output layer, compiles the model with the given optimizer, loss function, and metrics to optimize for,
    and fits the model on the training data for a specified number of epochs.  Finally, tests the model on the testing data
    and prints loss and accuracy.
    """

    def finish_model(self):
        self.model.add(tf.keras.layers.Dense(self.CLASSIFICATIONS, activation=tf.nn.softmax))
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        # Checkpoint the model, so it saves at each epoch

        checkpoint = ModelCheckpoint('models/' + self.model_name(), monitor='val_acc', verbose=1)
        callbacks_list = [checkpoint]

        self.model.fit(self.training_images, self.training_labels, epochs=self.epochs,
                           callbacks=callbacks_list)

    """Saves the model with a generated name in the models folder."""

    def save_model(self):
        # TODO: if there's not a models folder, make one

        self.model.save('models/' + self.model_name())

    def model_name(self):
        model_name = self.optimizer + '_' + self.loss + '_'
        for metric in self.metrics:
            model_name += metric
            model_name += '_'
        model_name += self.activation.__name__ + '_' + str(self.dense_layers) + "dense_layers_"
        model_name += str(self.epochs) + 'epochs'
        return model_name


class DataPrinter:
    def print_training_data(self, model_gen):
        training_loss, training_acc = model_gen.model.evaluate(model_gen.training_images,
                                                               model_gen.training_labels)
        print('training_loss, = ', training_loss, 'training_acc = ', training_acc)

    def print_validation_data(self, model_gen):
        val_loss, val_acc = model_gen.model.evaluate(model_gen.validation_images,
                                                     model_gen.validation_labels)
        print('val_loss, = ', val_loss, 'val_acc = ', val_acc)

    def print_testing_data(self, model_gen):
        test_loss, test_acc = model_gen.model.evaluate(model_gen.test_images, model_gen.test_labels)
        print('test_loss = ', test_loss, 'test_accuracy = ', test_acc)


if __name__ == '__main__':
    d = DataPrinter()
    model_gen1 = EmnistModelGenerator()
    model_gen1.save_model()
    # d.print_validation_data(model_gen1)
    # d.print_testing_data(model_gen1)
