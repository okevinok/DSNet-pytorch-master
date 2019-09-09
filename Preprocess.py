# %%
import h5py
import scipy.io as io
import PIL as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM


# from image import *
# %%
def gaussian_filter_density(gt):
    # Generates a density map using Gaussian filter transformation

    density = np.zeros(gt.shape, dtype=np.float32)

    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    # FInd out the K nearest neighbours using a KDTree

    pts = np.array(list(zip(np.nonzero(gt)[1].ravel(), np.nonzero(gt)[0].ravel())))
    leafsize = 2048

    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point

        # Convolve with the gaussian filter
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


# %%
root = 'data'
# %%
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')
path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]
# %%
# List of all image paths

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print(len(img_paths))
print(img_paths[0:5])
# %%
# test block 请不要执行
img_path = 'data\\part_B_final/train_data\\images\\IMG_328.jpg'

# 加载mat标签
mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
gt = mat["image_info"][0, 0][0, 0][0]
# print(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
print(mat)
print(mat["image_info"][0, 0][0, 0][0])  # 图片对应的信息 即 人头所在位置
print(gt.shape)  # 27张人脸的坐标

# 给对应的坐标位置进行标点为1
img = plt.imread(img_path)
img_groundtruth = np.asarray(img)
plt.imshow(img_groundtruth, cmap=CM.jet)

k = np.zeros((img.shape[0], img.shape[1]))
for i in range(0, len(gt)):
    if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
        k[int(gt[i][1]), int(gt[i][0])] = 1

# %% generate density map
k = gaussian_filter_density(k)  # 对图片进行高斯滤波

# %%
print(k)
groundtruth = np.asarray(k)
plt.imshow(groundtruth, cmap=CM.jet)
print("Sum = ", np.sum(groundtruth))

# %%
from tqdm import tqdm
from PIL import Image

i = 0
for img_path in tqdm(img_paths[:]):

    # Load sparse matrix 加载图片对应的mat文件，内含lable
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
    # gt才是 label文件 ，gt是n行两列的二维数据
    gt = mat["image_info"][0, 0][0, 0][0]

    # Read image
    img = plt.imread(img_path)

    # Create a zero matrix of image size
    k = np.zeros((img.shape[0], img.shape[1]))

    # Generate hot encoded matrix of sparse matrix
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1

    # generate density map
    k = gaussian_filter_density(k)
    image = Image.fromarray(255 * k)
    # image.show()

    # File path to save density map
    file_path = img_path.replace('.jpg', '.h5').replace('images', 'ground')

    with h5py.File(file_path, 'w') as hf:
        hf['density'] = k

# %%
path_test = "G:/pycharm/CSRnet-master/data/part_A_final/train_data/ground/IMG_107.h5"

# file_path = img_paths[-5].replace('.jpg','.h5').replace('images','ground')

file_path = path_test
print(file_path)

# %%
# Sample Ground Truth
gt_file = h5py.File(file_path, 'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth, cmap=CM.jet)
print("Sum = ", np.sum(groundtruth))
# %%
# Image corresponding to the ground truth
img = Image.open(file_path.replace('.h5', '.jpg').replace('ground', 'images')).convert('RGBA')
plt.imshow(img)
print(file_path.replace('.h5', '.jpg').replace('ground', 'images'))


