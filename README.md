# 02458_project
DTU 02458 all projects

## 1. How to save and read a npz(Numpy zip file)?
```
import numpy as np
test=np.load('./bvlc_alexnet.npy',encoding = "latin1")  #加载文件
doc = open('1.txt', 'a')  #打开一个存储文件，并依次写入
print(test, file=doc)  #将打印内容写入文件中

import numpy as np

# .npy文件是numpy专用的二进制文件
arr = np.array([[1, 2], [3, 4]])

# 保存.npy文件
np.save("../data/arr.npy", arr)
print("save .npy done")

# 读取.npy文件
np.load("../data/arr.npy")
print(arr)
print("load .npy done")

import numpy as np  
a = np.load("speech-linear-13100.npy") 
print(a)  
print("数据类型",type(a))           #打印数组数据类型
————————————————
版权声明：本文为CSDN博主「2020-canyang」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_37715669/article/details/89675551
```
