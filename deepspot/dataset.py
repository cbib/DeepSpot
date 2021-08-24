import os

import numpy as np
import skimage.io as io


class Dataset:
    def __init__(self, path_list):
        self.path_train = path_list[0]
        self.path_train_labels = path_list[1]
        self.path_test = path_list[2]
        self.path_test_labels = path_list[3]

        self.train_list = list_files(self.path_train)
        self.train_list_labels = list_files(self.path_train_labels)
        self.test_list = list_files(self.path_test)
        self.test_list_labels = list_files(self.path_test_labels)

        def get_images(path_list):
            dataset = []
            for file in path_list:
                try:
                    img = io.imread(file)
                    img = np.array(img, dtype=np.float32)
                    assert np.amax(img) > 0
                    assert img.shape[0] == 256
                    assert img.shape[1] == 256
                    dataset.append((np.array(img) / np.amax(img)).reshape(256, 256, 1))

                except Exception as e:
                    print_red("Image {} not found.\n{}".format(file, e))
            assert len(dataset) != 0
            assert np.amax(dataset[0]) > 0
            return np.array(dataset)

        self.dataset = get_images(self.train_list)
        self.dataset_labels = get_images(self.train_list_labels)
        self.dataset_test = get_images(self.test_list)
        self.dataset_test_labels = get_images(self.test_list_labels)

        assert len(self.test_list) == len(self.dataset_test)


def list_files(path):
    return [
        path + "/" + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    ]


def print_red(skk):
    print("\033[91m{}\033[00m".format(skk))


def print_gre(skk):
    print("\033[92m{}\033[00m".format(skk))

