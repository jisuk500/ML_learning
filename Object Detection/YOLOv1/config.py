# -*- coding: utf-8 -*-

#%% configs


VOCdatasetConfig= {
    'rootDir': "D:\GitHubRepos\ML_learning\VOC2007",
    'imageFolder': "JPEGImages",
    'imageExtension': ".jpg",
    'annotationFolder': "Annotations",
    'annotationExtension': ".xml",
    'trainCasesPath': "ImageSets\Segmentation\\train.txt",
    'valCasesPath' : "ImageSets\Segmentation\\val.txt",
    
    'textFileFolder': "Dataset"
    }


classNames =  {
    "aeroplane":0, 
    "bicycle": 1, 
    "bird": 2, 
    "boat":3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable":10,
    "dog": 11,
    "horse": 12,
    "motorbike":13 ,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
    }

YOLOConfig = {
    'inputSize': [448, 448],
    'outputGrid': [7, 7],
    'batchSize' : 16,
    
    'frontNet' : "VGG16"
    # ResNet50 / VGG16
    }

