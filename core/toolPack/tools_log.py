# --*-- coding: utf-8 --*--
# tools_log.py
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben  2020/4/13 上午11:28
import logging
import os
import math
from . import tools_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')

def print_list(datas):
    if type(datas) is dict:
        for k in datas:
            print(k, datas[k])
    else:
        for data in datas:
            print(data)

def print_args(args, logger=None):
    log = '\n' + '=' * 50 + '\n'
    args_dict = vars(args)
    for k in args_dict:
        log += f'{k}: {args_dict[k]}\n'
    log += '=' * 50 + '\n'
    if logger:
        logger.info(log)
    else:
        print(log,flush=True)

def print_line(name, maxLen=50):
    print()
    if len(name) < maxLen:
        left = math.floor((maxLen - len(name)) / 2.0)
        right = maxLen - left - len(name)
        name = '=' * left + name + '=' * right
    else:
        name = '=' + name
    print(name)
    print()

def createLogger(name='log', folder=None, quiet=False):
    if folder:
        logger = logging.getLogger(name)
        logger.propagate = False
        tools_file.createFolder(folder)
        fh = logging.FileHandler(os.path.join(folder, 'log.txt'))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    else:
        logger = None
    logger = __SystemLogger(logger=logger, quiet=quiet)
    return logger

class __SystemLogger():
    def __init__(self, logger=None, quiet=False):
        self.logger = logger
        self.quiet = quiet

    def info(self, msg, *args, nextLine=True, **kwarg):
        if not self.quiet:
            if self.logger:
                self.logger.info(msg, *args, **kwarg)
            else:
                if nextLine:
                    print(msg, flush=True)
                else:
                    print(f'\r{msg}', end='', flush=True)

    def warn(self, msg, *args, **kwarg):
        if not self.quiet:
            if self.logger:
                self.logger.warn(msg, *args, **kwarg)
            else:
                print(msg, flush=True)

    def warning(self, msg, *args, **kwarg):
        self.warn(msg, *args, **kwarg)

    def exception(self, msg, *args, **kwarg):
        if self.logger:
            self.logger.exception(msg, *args, **kwarg)
        else:
            print(msg, flush=True)

    def error(self, msg, *args, **kwargs):
        if self.logger:
            self.logger.error(msg, *args, **kwargs)
        else:
            print(msg, flush=True)

    def shutdown(self):
        if self.logger:
            for hdlr in self.logger.handlers:
                try:
                    hdlr.acquire()
                    hdlr.flush()
                    hdlr.close()
                except (OSError, ValueError):
                    pass
                finally:
                    hdlr.release()
