import numpy as np
from urllib import request
import gzip
import pickle
from sklearn.utils import shuffle
import pandas as pd
import os

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


class MNIST:

    def download_mnist(self, dirs="data/MNIST/"):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) + '/'
        
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], dirs+name[1])
        print("Download complete.")

    def save_mnist(self, dirs="data/MNIST"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) + '/'
        
        mnist = {}
        for name in filename[:2]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in filename[-2:]:
            with gzip.open(dirs + name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        with open(dirs + "mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    def init(self):
        self.download_mnist()
        self.save_mnist()

    def load(self, dirs="data/MNIST"):
        lib_dir = os.path.dirname(os.path.realpath(__file__))# + dirs
        dirs = os.path.join(lib_dir,dirs) + '/'

        with open(dirs + "mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


class CurveTest:

    def __init__(self):
        x_data = np.linspace(0, 10, 300)[:, np.newaxis]
        noise = np.random.normal(-0.05, 0.05, x_data.shape)
        y_data = 2 * np.sin(x_data) + np.cos(2 * x_data - 3) + 3 * np.log(x_data + 0.5) - 4
        self.x_data = x_data / 10
        self.y_data = y_data / 10 + noise
        # self.x_data, self.y_data = shuffle(self.x_data, self.y_data)
        self.len = len(self.x_data)

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.len
        xx, yy = shuffle(self.x_data, self.y_data)
        return xx[:batch_size], yy[:batch_size]


class CurveTest2:

    def __init__(self):
        # DATASET from file
        df = pd.read_csv('dataset_1.txt', sep=',')
        # print(df.columns)
        df = df.drop('C', axis=1)
        self.x_data = np.array(df['X'].values).reshape(-1, 1)
        self.y_data = np.array(df['Y'].values).reshape(-1, 1)
        # self.x_data, self.y_data = shuffle(self.x_data, self.y_data)
        self.len = len(self.x_data)

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.len
        xx, yy = shuffle(self.x_data, self.y_data)
        return xx[:batch_size], yy[:batch_size]


# class Iris:
#
#     def __init__(self):
#
