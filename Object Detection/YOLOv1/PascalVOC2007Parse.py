# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:36:20 2020

@author: jisuk
"""

#%% import module

import os
import sys
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
import tensorflow as tf

import config as cfg

#%% classes

class DatasetTool:
    def __init__(self):
        self.imagesPath = os.path.join(cfg.VOCdatasetConfig['rootDir'], cfg.VOCdatasetConfig['imageFolder'])
        
        self.textFileFolder = cfg.VOCdatasetConfig['textFileFolder']
        self.trainCasesPath = cfg.VOCdatasetConfig['trainCasesPath']
        self.valCasesPath = cfg.VOCdatasetConfig['valCasesPath']
        
        
    def splitDataset(self,val = 0.2):
        ann_root, ann_dir, ann_files = next(os.walk(self.imagesPath))
        
        i = 0
        valBatch = 1/ val
        trainText = open(os.path.join(self.textFileFolder,"trainNames.txt"),'w+')
        valText = open(os.path.join(self.textFileFolder,"valNames.txt"),'w+')
        for file in ann_files:
            file = file.split('.')
            file = file[0]
            
            if(i > valBatch):
                i = 0
                valText.write(file+"\n")
            else:
                trainText.write(file+"\n")
            
            i = i + 1
        
        trainText.close()
        valText.close()
            

class PascalVOC2007:
    
    def __init__(self):
        
        self.RootPath = cfg.VOCdatasetConfig['rootDir']
        self.imageFolderPath = cfg.VOCdatasetConfig['imageFolder']
        self.imageExtension = cfg.VOCdatasetConfig['imageExtension']
        self.annotationFolderPath = cfg.VOCdatasetConfig['annotationFolder']
        self.annotationExtension = cfg.VOCdatasetConfig['annotationExtension']
        
        self.imageFolderPath = os.path.join(self.RootPath,self.imageFolderPath)
        self.annotationFolderPath = os.path.join(self.RootPath,self.annotationFolderPath)
        
        self.classNames = cfg.classNames
        self.textFileFolder = cfg.VOCdatasetConfig['textFileFolder']
        
        self.trainCasesPath = cfg.VOCdatasetConfig['trainCasesPath']
        self.valCasesPath = cfg.VOCdatasetConfig['valCasesPath']
        
    
    def getImagePath(self, imageName):
        return os.path.join(self.annotationFolderPath, imageName+self.imageExtension)
    
    
    def getAnnotationPath(self, imageName):
        return os.path.join(self.annotationFolderPath,imageName+self.annotationExtension)
    
    
    def getAnnotation(self, annotationPath, pathtype):
        if(pathtype == 'fileName'):
            annotationPath = self.getAnnotationPath(annotationPath)
        
        result = dict()
        
        xml = open(annotationPath,'r')
        tree = Et.parse(xml)
        root = tree.getroot()
    
        result['filename'] = root.find("filename").text
    
        size = root.find("size") 
        result['width'] = size.find("width").text
        result['height'] = size.find("height").text
        result['channels'] = size.find("depth").text
        
        
        objects = root.findall("object")
        result['objects'] = []
        for obj in objects:
            
            objInfo = dict()
            
            objInfo['name'] = obj.find("name").text
        
            bndbox = obj.find("bndbox")
            objInfo['xmin'] = bndbox.find("xmin").text
            objInfo['ymin'] = bndbox.find("ymin").text
            objInfo['xmax'] = bndbox.find("xmax").text
            objInfo['ymax'] = bndbox.find("ymax").text
            
            result['objects'].append(objInfo)
        
        return result
    
    
    def XML2StrLine(self, objDict):
        string = objDict['filename']
        
        for bndbox in objDict['objects']:
            string = string +" " + str(bndbox['xmin']) + " " + str(bndbox['ymin']) + " " + str(bndbox['xmax']) + " " + str(bndbox['ymax']) + " " + str(self.classNames[bndbox['name']])
        
        return string
    
    def MakeTextFiles(self):
        trainNames = open(os.path.join(self.textFileFolder,"trainNames.txt"),'r')
        trainText = open(os.path.join(self.textFileFolder,"train.txt"),'w+')
        
        for line in trainNames.readlines():
            line = line.strip().split(' ')
            
            filename = line[0]
            annotationDict = self.getAnnotation(filename,'fileName')
            string = self.XML2StrLine(annotationDict)
            trainText.write(string+"\n")
            
        trainText.close()
        
        
        valNames = open(os.path.join(self.textFileFolder,"valNames.txt"),'r')
        valText = open(os.path.join(self.textFileFolder,"val.txt"),'w+')
        
        for line in valNames.readlines():
            line = line.strip().split(' ')
            
            filename = line[0]
            annotationDict = self.getAnnotation(filename,'fileName')
            string = self.XML2StrLine(annotationDict)
            valText.write(string+"\n")
            
        valText.close()
    
    

#%% main code

#d = DatasetTool()
#d.splitDataset(val=0.2)

            
            
            
    
    
    
    
    
        
        