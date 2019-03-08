from keras.utils import np_utils
import tensorflow as tf
import numpy as np
import utils
from keras.preprocessing.image import ImageDataGenerator
from cNNetwork import CNNetwork

def prepareData():
        # Load the data and allocate it into correct matrices. CIFAR10 is a dataset supported by Keras.
        # http://www.cs.toronto.edu/~kriz/cifar.html
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        categories = 10

        # Ensuring that the values are in 32 decimal float, which is desired for values 0<x<1
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        # Z-score normalization
        mean = np.mean(x_train)
        std = np.std(x_train)
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)

        # Allocate validation data into categories to be used with crossentropy
        y_train = np_utils.to_categorical(y_train, categories)
        y_test = np_utils.to_categorical(y_test, categories)
        print(x_train.shape[1:])

        return (x_train, y_train, x_test, y_test)

def createNetwork():
        x_train, y_train, x_test, y_test = prepareData()

        batch_size = 64  # no. of training examples per iteration
        epochs = 20  # no. of iterations the neurons update their weights
        categories = 10  # CIFAR10 has 10 categories
        img_dimensions = 32  # image dimensions (n*n)

        # Create a new instance of CNNetwork class, which creates the network model and image data generator
        cifar10 = CNNetwork(batch_size, categories, img_dimensions, x_train.shape[1:])
        model = cifar10.model
        model.summary()  # Show a table containing details of the created neural network
        runNetwork(model, batch_size, epochs)

def runNetwork(model, batch_size, epochs):
        x_train, y_train, x_test, y_test = prepareData()
        # Compile the network
        model.compile(loss="categorical_crossentropy",  # use the loss function to take into note the probabilities of each category
                optimizer="adam",                       # adaptive moment estimation
                metrics=["accuracy"])                   # use accuracy as a metric of progress during training

        # Training
        model.fit_generator(createGenerator().flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                            validation_data=(x_test, y_test))

        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("Accuracy: ", str(test_acc))
        utils.saveModel(model)

def createGenerator():   # Generator to process images and apply different methods on them to normalize data
        generator = ImageDataGenerator(
            rotation_range=15,          # Apply random rotation to images, from 0 to 180 degrees
            width_shift_range=0.2,      # Randomly shift images horizontally, portion of the total width
            height_shift_range=0.2,     # -||- vertically
            horizontal_flip=True,       # Randomly flip images horizontally
        )

        return generator