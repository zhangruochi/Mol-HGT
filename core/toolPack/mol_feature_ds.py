# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben 2020/5/26 下午3:15
from descriptastorus.descriptors import rdNormalizedDescriptors
import numpy as np

def getFeature_ds(smi):
    try:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        out = generator.process(smi)
        if out[0]:
            feature = np.array(out[1:], dtype=np.float32)
            feature = np.nan_to_num(feature)
            return feature
    except:
        pass
