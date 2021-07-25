import sys
import os
import argparse
from .tools_file import createDirFolder

def run_bg():
    name0 = '--bg'
    name1 = '--bgt'
    if name0 in sys.argv:
        pos = sys.argv.index(name0)
        name = name0
    elif name1 in sys.argv:
        pos = sys.argv.index(name1)
        name = name1
    else:
        return
    if pos + 1 == len(sys.argv):
        sys.argv.pop(pos)
        fileLog = 'log'
    else:
        fileLog = sys.argv[pos + 1]
        sys.argv.pop(pos)
        sys.argv.pop(pos)
    if os.path.exists(fileLog):
        os.remove(fileLog)
    createDirFolder(fileLog)
    os.system(f'nohup python {" ".join(sys.argv)} >{fileLog} 2>&1 &')
    if name == name1:
        os.system(f'tail -f {fileLog} -n 10000')
    exit(0)

def optionParse(name, default, choices=None):
    if name in sys.argv:
        id = sys.argv.index(name)
        option = sys.argv[id + 1]
    else:
        option = default
    if choices is not None and option not in choices:
        raise Exception(f'{option} of {name} is not in {choices}')
    ap = argparse.ArgumentParser()
    ap.add_argument(name, type=str, default=option)
    return option, ap
