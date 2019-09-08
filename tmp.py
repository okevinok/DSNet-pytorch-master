import json

import os, glob
#
# root = 'data/'
# #%%
# #now generate the ShanghaiA's ground truth
# part_A_train = os.path.join(root,'part_A_final/train_data','images')
# part_A_test = os.path.join(root,'part_A_final/test_data','images')
# part_B_train = os.path.join(root,'part_B_final/train_data','images')
# part_B_test = os.path.join(root,'part_B_final/test_data','images')
# path_sets = [part_B_train]
# #%%
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
#
# print(img_paths)
# print(len(img_paths))
#
# with open("part_B_train_wlq.json","w", encoding="utf-8") as file:
#     json.dump(img_paths, file)

