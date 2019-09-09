import numpy as np
from torch.utils.data import Dataset
from math import *
from PIL import Image
import h5py
import cv2
import torch

def create_img(path):
    # Function to load,normalize and return image
    im = Image.open(path).convert('RGB')
    im = np.array(im)
    im = im / 255.0

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    # print(im.shape)
    # im = np.expand_dims(im,axis  = 0)
    return im


def get_input(path):
    img = create_img(path)

    return np.array(img)


def get_output(path):
    # import target
    # resize target

    gt_file = h5py.File(path, 'r')
    target = np.asarray(gt_file['density'])
    img = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64
    img = np.expand_dims(img, axis=3)

    # print(img.shape)

    return np.array(img)

import os
import glob

root = 'data'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')


class Generate_data(Dataset):
    def __init__(self,file_path=[part_A_train]):
        self.path_sets = file_path
        self.img_paths = []
        self.length = 0
        self.get_data_path()
        self.x = None
        self.y = None

    def __getitem__(self, index):
        self.x = get_input(self.img_paths[index])
        self.y = get_output(self.img_paths[index].replace(".jpg", ".h5").replace("images", "ground"))
        return self.x, self.y

    def get_data_path(self):
        for path in self.path_sets:
            for img_path in glob.glob(os.path.join(path, '*.jpg')):
                self.img_paths.append(str(img_path))

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import cm as c

    data = Generate_data()
    print(data.x.shape)
    plt.imshow(data.x.reshape(data.x.shape[0],data.x.shape[1],data.x.shape[2]))
    plt.show()

    print(data.y.shape)
    plt.imshow(data.y.reshape(data.y.shape[0], data.y.shape[1]), cmap=c.jet)
    plt.show()

    # TODO Dataloader 没有运行成功 明天 具体了解dataloader使用

