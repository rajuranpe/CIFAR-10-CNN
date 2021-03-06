from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os
import createNetwork
import sys

# save the model and weights
def saveModel(model):
    json = model.to_json()  # convert the model into .json
    with open("cifar10_model.json", "w") as json_file:
        json_file.write(json)
    model.save_weights("cifar10_model.h5")  # save the weights to HDF5 format

# convert the model from .json file and apply the weight from HDF5 into it
def loadModel():
    try:
        json = open("cifar10_model.json", "r")
        model_json = json.read()
        json.close()
        model = model_from_json(model_json)
        model.load_weights("cifar10_model.h5")
        return model
    except FileNotFoundError:
        print("No saved models yet! Please create a new network.")
        sys.exit()

# Function for numbers == categories
def categoryName(n):
    category = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    type = category.get(n)
    return type

# Visualize a small preview of evaluated predictions and their true values in Jupyter notebook or similiar app
def preview(model, img_dimensions, index):
    # @require index <= len(x_test) - 11
    x_train, y_train, x_test, y_test = createNetwork.prepareData()
    fig = plt.figure(figsize=(50, 50))
    for i in range(0, 10):
        image_index = index + i
        sub = fig.add_subplot(10, 1, i + 1)
        pred = model.predict(x_test[image_index].reshape(1, img_dimensions, img_dimensions, 3))
        y_labels = y_test.astype(int)
        title = "Subject of the picture is probadly " + categoryName(pred.argmax()) + " in reality, it is " + categoryName(y_labels[image_index].argmax())
        sub.set_title(title)
        sub.imshow(x_test[image_index].reshape(img_dimensions, img_dimensions, 3), interpolation='nearest')
