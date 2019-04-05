"""
@author:  wangquaxiu
@time:  2018/9/27 10:36
"""
import os, sys
from PIL import Image
from numpy import unicode

"""
将filePath文件下的图片保存在newFilePath文件夹下的相应子文件夹中
pic 是字典，存放每个图片要移到的子文件夹名
"""
def moveImg(filePath, newFilePath):
    #     filePath = unicode(filePath, 'utf-8')
    # newFilePath = unicode(newFilePath, 'utf-8')

    print(filePath, newFilePath)
    img = Image.open(filePath)
    img.save(newFilePath)
