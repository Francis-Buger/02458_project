
import pandas as pd
import numpy as np
import tabulate
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import pdb

# filename=r'/content/drive/Othercomputers/我的 MacBook Pro/Desktop/02458/results.txt'
# filename=r'/Users/baixiang/Desktop/02458/results.csv'
filename = '/Users/francis/Desktop/DTU/classes/02458/Stimulus presentation script/results.txt'
attributeNames = ['subject_id','trial_number','stimulus_filename','answer','reaction_time']
result = ['Not smiling','Smiling']
result_dits = [0, 1]
df = pd.read_csv(filename, names = attributeNames)
train_data = df.values

mean = train_data[:,4].mean()                   # caluate the mean of the data we collected
std  = train_data[:,4].std()                    # caluate the Standard deviation of the data we collected

# 看有多少个测试者
observer_ID = sorted(set(train_data[:,0]))      # abstract the observer's ID


N = len(observer_ID)                            # count the amount of observer involved in our experiment

# define the outlier threshod
upper_threshod, lower_threshod = 1, 0.2

'''clean the data, weed out outliers'''
X = []  # empty list to save the cleaned data

# counter of outliers
outliter = 0
for i in range(len(train_data[:,0])):
  if  lower_threshod < train_data[i,4] < upper_threshod:
    X.append(train_data[i])
  else:
    outliter += 1
# print('There are', outliter, 'outliers in our data')

# '''show fist 10 rows of cleaned data X'''
# display(HTML(tabulate.tabulate(X[:9], headers=attributeNames, showindex='always', tablefmt='html')))


'''coding the face emotion results smelling and not smelling to digits'''
X = np.array(X)
for i in range(len(X[:,0])):
  if X[i,3] == 'Smiling':
    X[i,3] = 1
  else:
    X[i,3] = 0

# '''show fist 10 rows of cleaned data X'''
# display(HTML(tabulate.tabulate(X[:9], headers=attributeNames, showindex='always', tablefmt='html')))

plt.rcParams['figure.dpi'] = 200  # setting the 
# names = locals()
names = {}
for i in range(N):
  names['observer' + str(i)] = []


for j in range(len(X[:,0])):
  for i in range(N):
    # Zhijian Feng 21.30.21.05
    # if X[j,0] == observor_ID[i]:
    if X[j,0] == observer_ID[i]:
      names['observer' + str(i)].append(X[j])

# convert the list to numpy array and Normalization
for i in range(N):
  names['observer' + str(i)] = np.array(names['observer' + str(i)])
  tmp_mean = names['observer' + str(i)][:,4].mean()
  tmp_std = names['observer' + str(i)][:,4].std()
  temp_normal = ((names['observer' + str(i)][:,4]- tmp_mean) / tmp_std)
  temp_normal = np.reshape(temp_normal,(len(temp_normal),1))
  names['observer' + str(i)] = np.hstack((names['observer' + str(i)],temp_normal))

# print(names['observer0'][:,1])
# 现在整个数据结构就是，names这个dict用来存所有的数据，里面用 “observer$num.format(test_number,%d)" 来索引每一个用户。
# 然后里面每一个observer里面的结构是一个二维数组，保持原来的特性 
# 二维数组的属性：subject_id, trial_number, stimulus_filename, answer, reaction_time, normal_value

# for i in range(N):
#   # names['observer' + str(i)] = np.array(names['observer' + str(i)])
#   # observer_mean = names['observer' + str(i)][:,4].mean()
#   # observer_std = names['observer' + str(i)][:,4].std()
#   # names['observer' + str(i) + '_normal'] = ((names['observer' + str(i)][:,4]- observer_mean) / observer_std).flatten()
#   # print(names['observer' + str(i) + '_normal'].shape)
#   # print(names['observer' + str(i) + '_normal'].flatten())
#   plt.subplot(1, N, i+1)
#   plt.hist(names['observer' + str(i)][:,5], bins=20)
#   # plt.title('observer' + str(i) + '[' + str(observer_ID[i]) + ']')
#   plt.title('observer' + str(i))
#   plt.xlabel('ReactTimeSpan')
#   plt.ylabel('Distribution')
#   plt.tight_layout(pad=0.4, w_pad=1, h_pad=1.0)


import cv2
import os
import pdb
import scipy
from sklearn import datasets, linear_model
import scipy.linalg as linalg
import matplotlib.pyplot as plt

image_path = r'/Users/francis/Desktop/DTU/classes/02458/ARArchive/'

all_image_values = np.zeros((50,50),dtype=np.uint16)
# all_image_values = cv2.cvtColor(all_image_values, cv2.COLOR_GRAY2BGR)

ori_imgs = []
for n in range(len(names)):
  for i in range(len(names["observer" + str(n)])):
    # start
    for j in range(len(names["observer" + str(n)][:,2])):
      f_name = names["observer"+str(n)][j,2]
      tmp_path = os.path.join(image_path,f_name)
      img = cv2.imread(tmp_path,-1)
      # cv2.imshow("img",img)
      # cv2.waitKey(0)
      pdb.set_trace()
      ori_imgs.append(img)
      # 做一个累加
      all_image_values[:,:] = all_image_values[:,:] + img[:,:]

    ori_imgs = np.array(ori_imgs)
    # 均值
    mean_value = all_image_values / (len(os.listdir(image_path))-1)
    # 每一张图都减均值
    for i in range(len(ori_imgs)):
        ori_imgs[i] = ori_imgs[i] - mean_value

    # sequence
    # reshap每一张图 50*50
    sequence_imgs = []
    for i in range(len(ori_imgs)):
        sequence_imgs.append(ori_imgs[i].reshape(1,50*50))

    # 变成 样本*(每一张图 50*50): sample*[length*width]
    sequence_imgs = np.array(sequence_imgs)
    d_1,d_2,d_3 = sequence_imgs.shape
    sequence_imgs = np.reshape(sequence_imgs,(d_1,d_3))

    # 做PCA
    U,S,V = linalg.svd(sequence_imgs,full_matrices=True)
    V = V.T

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()
    threshold = 0.9
    ##
    # Plot variance explained 绘制可解释变异
    plt.figure()
    plt.plot(rho,'o-')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained value');


    # # Plot variance explained
    # plt.figure()
    # plt.plot(range(1,len(rho[:50])+1),rho[:50],'x-')                  # 绘制各在成分上投影的方差
    # plt.plot(range(1,len(rho[:50])+1),np.cumsum(rho[:50]),'o-')       # 绘制各成分累加的可解释变异或课解释方差
    # plt.plot([1,len(rho[:50])],[threshold, threshold],'k--')     # 绘制压缩要求阈值
    # plt.title('Variance explained by principal components')
    # plt.xlabel('Principal component')
    # plt.ylabel('Variance explained')
    # plt.legend(['Individual','Cumulative','Threshold'])
    # plt.grid()
    # plt.show()

    # 转换每一张图的shape：17*17
    Z = sequence_imgs @ V[:,:289]
    pca_datas = []
    for i in range(len(Z)):
      pca_datas.append(np.reshape(Z[i],(17,17)))
    answer = names["observer"+str(n)][:,3]

    # regr = linear_model.LinearRegression()
    # regr.fit(pca_datas, answer)

# # Plot variance explained
# plt.figure()
# plt.plot(rho[:8],'o-')
# plt.title('Variance explained by principal components');
# plt.xlabel('Principal component');
# plt.ylabel('Variance explained value');

# # Plot PCA of the data
# f = plt.figure()
# plt.title('pixel vectors of handwr. digits projected on PCs')
# n = [0,1,2,3]
# for c in n:
#     # select indices belonging to class c:
#     class_mask = (y == c)
#     plt.plot(Z[class_mask,0], Z[class_mask,1], 'o')
# plt.legend("classNames")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
