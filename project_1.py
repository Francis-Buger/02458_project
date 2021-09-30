import pandas as pd
import numpy as np
import os
from PIL import Image

f_path = '/Users/francis/Desktop/DTU/classes/02458/Stimulus presentation script/results.txt'
photo_path = '/Users/francis/Desktop/DTU/classes/02458/ARArchive/'
# print(os.path)
# 4.a 
content = pd.read_csv(f_path)
np_content = content.to_numpy()

# 4.b
n_resp_time = content['resp_time'].to_numpy()
idx = np.where(n_resp_time < 0.2)
content.drop(index=idx[0])

# 4.c
n_test_name = content['test_id'].to_numpy()
n_test_num = np.unique(n_test_name)

split_by_name = []
for i in range(len(n_test_num)):
    split_by_name.append(content[content['test_id'] == n_test_name[i]])

n_resp_time = content['resp_time'].to_numpy()

# 4.d
content['idx'] = range(0, len(n_resp_time))
all_pixel = np.zeros((50,50))
for file_ in os.listdir(photo_path):
    tmp_path = os.path.join(photo_path,file_)
    n_img = np.array(Image.open(tmp_path))
    # print(n_img.shape)
    # break
    all_pixel = all_pixel + n_img

print(all_pixel.shape)
    



# print(content)





