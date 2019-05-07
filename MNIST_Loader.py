# Code adapted from GitHub user hsjeong5 -- https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py

import numpy as np
import gzip
import pickle

filename = [
["training_images","emnist-balanced-train-images-idx3-ubyte.gz"],
["test_images","emnist-balanced-test-images-idx3-ubyte.gz"],
["training_labels","emnist-balanced-train-labels-idx1-ubyte.gz"],
["test_labels","emnist-balanced-test-labels-idx1-ubyte.gz"]
]


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    save_mnist()

def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    #here, we need to divide training into training and 5% validation
    validation_images = mnist["training_images"][107160:]
    validation_labels = mnist["training_labels"][107160:]
    training_images = mnist["training_images"][:107159]
    training_labels = mnist["training_labels"][:107159]


    return (training_images, training_labels), (mnist["test_images"], mnist["test_labels"]), (validation_images,validation_labels)

if __name__ == '__main__':
    init()
    load()