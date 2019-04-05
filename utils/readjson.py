"""
@author:  wangquaxiu
@time:  2018/9/27 10:48
"""

import json

def resolveJson(path):
    file = open(path, encoding='utf-8')
    fileJson = json.load(file)
    return fileJson


#
#
# for i in range(1, len(train_json)):
#     print(train_json[i].get("image_id"), train_json[i].get("disease_class"))