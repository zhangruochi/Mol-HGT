# --*-- coding: utf-8 --*--
# Copyright (c) 2019 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben  2021/7/5

import json
import pickle
import pandas as pd
from filelock import FileLock
from .tools_file import *
import logging

logging.getLogger('filelock').setLevel(logging.ERROR)

# ======================pickle=======================
def pkl_write(path, variable):
    createDirFolder(path)
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

class Pkl_write_list():
    def __init__(self, path, bDelFile=True):
        self.path = path
        self.path_log = self.path + '.log'
        self.count = 0
        if not bDelFile and os.path.exists(self.path) and os.path.exists(self.path_log):
            log = dict_read(self.path_log)
            self.count += log['count']
        else:
            createFile(path, bDelFile=bDelFile)
        self.f = open(self.path, 'ab')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dict_write(self.path_log, {'count': self.count})
        self.f.close()

    def save(self, data):
        pickle.dump(data, self.f)
        self.count += 1

    def save_list(self, datas):
        for data in datas:
            self.save(data)

def pkl_read(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
                return data
            except Exception as e:
                return None

def pkl_read_list(path):
    with open(path, 'rb') as f:
        while (True):
            try:
                data = pickle.load(f)
                yield data
            except:
                break

class Pkl_read_list():
    def __init__(self, filePath):
        self.filePath = filePath
        self.logPath = self.filePath + '.log'
        log = dict_read(self.logPath)
        if log is not None:
            self.count = log['count']
        else:
            self.count = None

    def generator(self):
        return pkl_read_list(self.filePath)

# ======================dict=======================
def dict_read(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        data = dict(zip(df['key'], df['value']))
        return data

def dict_write(path, data, update=False):
    if update and os.path.exists(path):
        try:
            datas_old = dict_read(path)
            datas_old.update(data)
            data = datas_old
        except:
            pass
    df = pd.DataFrame()
    df['key'] = list(data.keys())
    df['value'] = list(data.values())
    df.to_csv(path, index=False)

# ======================json=======================
def json_write(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def json_read(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# ======================lock=======================
class Lock_io():
    def __init__(self, path, lock='file'):
        self.path = path
        if lock == 'file':
            path_lock = self.path + '.lock'
            createFile(path_lock)
            self.lock = FileLock(path_lock)
        else:
            self.lock = lock

    def read(self):
        pass

    def write(self, data):
        pass

class Lock_io_dict(Lock_io):
    def __init__(self, path, lock='file'):
        super().__init__(path, lock=lock)

    def read(self):
        if self.lock is None:
            data = dict_read(self.path)
        else:
            with self.lock:
                data = dict_read(self.path)
        return data

    def write(self, data, update=False):
        if self.lock is None:
            dict_write(self.path, data, update=update)
        else:
            with self.lock:
                dict_write(self.path, data, update=update)

class Lock_io_pkl(Lock_io):
    def __init__(self, path, lock='file'):
        super().__init__(path, lock=lock)

    def read(self):
        if self.lock is None:
            data = pkl_read(self.path)
        else:
            with self.lock:
                data = pkl_read(self.path)
        return data

    def write(self, data):
        if self.lock is None:
            pkl_write(self.path, data)
        else:
            with self.lock:
                pkl_write(self.path, data)
