import numpy as np
import tensorflow as tf
from PIL import Image
import os, pickle, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def list_files_in_directory(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    # Because I often work on a Mac, I attempt to remove the .DS_Store file,
    # which really does no good for most practical purposes.
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        print('No DS_Store to remove')
    return file_list


# Sometimes you just want to unpack a pcl file into a csv
def pickle2csv(in_name, out_name):
    with open(in_name, 'rb') as pickle_file:
        numpy_array = pickle.load(pickle_file)

    np.savetxt(out_name, numpy_array)
    return 0


def save_1d_line_plot_as_png(x, y, title='title', saved_name='myplot'):
    fig = plt.figure()
    plt.plot(x, y)
    plt.title(title)
    fig.savefig(saved_name + ".png")
    plt.clf()

# Xavier Initialization
def xavier_init(fan_in, fan_out, constant=1):
    """Initialize network weights with Xavier Initialization"""
    low = -constant*np.sqrt(6/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def save_matrix_as_png(matrix, title='title', saved_name='mymatrix'):
    fig = plt.figure()
    # 

def save_matrix_as_csv(matrix, output_name):
    np.savetxt(output_name, matrix, delimiter=',')


class MNIST(object):
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.eval_data = mnist.test.images
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def next_batch(self, batch_size):
        train_indices = np.random.randint(self.train_data.shape[0], size=batch_size)
        return self.train_data[train_indices, :]

    def test_batch(self, batch_size):
        eval_indices = np.random.randint(self.eval_data.shape[0], size=batch_size)
        return self.eval_data[eval_indices, :]
