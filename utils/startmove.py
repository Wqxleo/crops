"""
@author:  wangquaxiu
@time:  2018/9/27 10:38
"""
import utils
from utils.move import moveImg
from utils.readjson import resolveJson

train_json = resolveJson("G:/Python/crops/data/train/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json")
for i in range(1, len(train_json)):
    print(train_json[i].get("image_id"), train_json[i].get("disease_class"))
    filePath = "G:/Python/crops/data/train/AgriculturalDisease_trainingset/images/"+str(train_json[i].get("image_id"))

    newFilePath = "G:/Python/crops/data/train/"+str(train_json[i].get("disease_class"))+'/'+str(train_json[i].get("image_id"));
    moveImg(filePath, newFilePath)

