# --*-- coding: utf-8 --*--
# Copyright (c) 2019 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben  2019/10/23 上午10:29

import os
import shutil
import time

def createFolder(folderPath, bDelFolder=False):
    if bDelFolder and os.path.exists(folderPath):
        shutil.rmtree(folderPath)
    if folderPath != '' and not os.path.exists(folderPath):
        os.makedirs(folderPath)

def createDirFolder(filePath):
    folder = os.path.dirname(filePath)
    createFolder(folder)

def createFile(filePath, bDelFile=True):
    if bDelFile:
        delFile(filePath)
    if not os.path.exists(filePath):
        createDirFolder(filePath)
        os.mknod(filePath)

# ==================================
def delFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

def delFolder(folderPath):
    if os.path.exists(folderPath):
        shutil.rmtree(folderPath)

# ==================================
def clearFileInFolder(folderPath):
    if os.path.exists(folderPath):
        ls = os.listdir(folderPath)
        for i in ls:
            c_path = os.path.join(folderPath, i)
            if not os.path.isdir(c_path):
                os.remove(c_path)

def clearSubFolders(folderPath):
    paths = getAllSubFolderPath(folderPath)
    for path in paths:
        shutil.rmtree(path)

# ==================================
def getAllFilePath(folder):
    names = os.listdir(folder)
    paths = []
    for name in names:
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            paths.append(path)
        else:
            paths.extend(getAllFilePath(path))
    return paths

def getAllSubFolderPath(folder):
    paths = []
    if os.path.exists(folder):
        names = os.listdir(folder)
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isdir(path):
                paths.append(path)
    return paths

def getAllSubFolderName(folder):
    names = os.listdir(folder)
    folderNames = []
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            folderNames.append(name)
    return folderNames

# ==================================
def checkFiles(filePaths):
    fails = []
    for path in filePaths:
        if not os.path.exists(path):
            print(f'not existed: {path}')
            fails.append(path)
    if len(fails) > 0:
        time.sleep(1)
        print('check again')
        checkFiles(fails)
