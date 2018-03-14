from PIL import Image
import numpy as np
import random, os

def directory_filelist(target_directory):
    file_list = [f for f in os.listdir(target_directory)
                 if os.path.isfile(os.path.join(target_directory, f))]
    try:
        file_list.remove('.DS_Store')
    except ValueError:
        print('good news everyone')
    return file_list

def load_tiff_layer(file_name, layer):
    img = Image.open(file_name)
    try:
        img.seek(layer)
    except EOFError:
        print('Not a valid layer')
    # The below is super not secure, so be careful
    vectorized_image = np.reshape(img, [np.prod(np.shape(img)),])
    return vectorized_image


def fetch_batch_of_tiff_layers(list_of_images, batch_size, layer):
    batch_list = random.sample(list_of_images, batch_size)
    array_of_image_vectors = np.asarray([load_tiff_layer(f, layer) for f in batch_list])
    return array_of_image_vectors

class TIFF_Stream():
    def __init__(self):
        target_directory = "/Volumes/My Book Duo/Research_Datasets/STEM4D_Data/s9_100nm_sto_4uc_12ucModel_dz_5nm_0p5_0_tilt/tiff/10nm/"
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]

    def next_batch(self, batch_size):
        batch_list = random.sample(self.full_filelist, batch_size)
        array_of_image_vectors = np.asarray([load_tiff_layer(f, 1) for f in batch_list])
        return array_of_image_vectors
