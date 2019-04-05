"""
@author:  wangquaxiu
@time:  2018/9/27 11:32
"""

import os

for i in range(0, 61):
    path = "G:/Python/crops/data/train/"+str(i)+"/"
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
         os.makedirs(path)      #makedirs 创建文件时如果路径不存在会创建这个路径
         print("---  new folder...  ---")
         print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

