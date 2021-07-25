import tensorflow as tf
from argparse import Namespace
from tensorflow.keras import layers
from hyperopt import hp
from ..layer.readout_vnt import Readout_vnt
from ..layer.gnnLayer_molHgt import GnnLayer_molHgt
from ..layer.update_gru import Update_gru
from ..tools_tfg import Output_block
from ..molPred_tfg import MolPred_tfg

config_default = \
    {
        'graphType': 0,
        'graphSteps': 3,
        'hiddenSize': 600,
        'heads': 10,
        'extFeat': 0,
        'modelEnsemble': 1,
        'fcEnsemble': 1,
        'dropout': 0.0,
        'vntResidual': 1,
        'epochs': 0,
        'cEpochs': 30,
        'rEpochs': 80,
        'warmupEpochs': 1,
        'lrStart': 0.0,
        'lrMax': 1e-4,
        'lrEnd': 1e-5,
        'lrCurveType': 'cos',
        'batchSize': 32,
        'batchMax': 'edge_3000',
        'earlyStop': 30,
        'cLossFn': 'crossentropy',
        'rLossFn': 'mae',
        'rYScale': 0,
    }

def getMolPred(args):
    return MolPred_tfg(args, Model_molHgt)

def getHpSpace():
    hpSpace = {
        'hiddenSize': hp.quniform('hiddenSize', low=200, high=800, q=40),
        'graphSteps': hp.quniform('graphSteps', low=2, high=4, q=1),
        'lrMax': hp.choice('lrMax', [0.0001, 0.0005])
    }
    return hpSpace

def addArgument_model(ap):
    for k in config_default:
        ap.add_argument(f'--{k}', type=type(config_default[k]), default=config_default[k])

class Model_molHgt(tf.keras.Model):
    def __init__(self, args: Namespace):
        super(Model_molHgt, self).__init__()
        # =================================
        self.nTypeNum = 11
        self.eTypeNum = 5
        self.extFeat = args.extFeat
        # =================================
        self.gnn_block = Gnn_block(args.hiddenSize, self.nTypeNum, self.eTypeNum, args.graphSteps,
                                   heads=args.heads, graphType=args.graphType, vntResidual=args.vntResidual)
        # =================================
        self.output_block = Output_block(labels=args.labels,
                                         taskType=args.taskType,
                                         hiddenSize=args.hiddenSize,
                                         extFeat=args.extFeat,
                                         dropout=args.dropout,
                                         yMean_train=args.yMean_train,
                                         yStd_train=args.yStd_train,
                                         rYScale=args.rYScale,
                                         fcEnsemble=args.fcEnsemble)

    def call(self, bg, training=False):
        nId = bg.nId
        nType = bg.nf[:, : self.nTypeNum]
        nType_id = tf.where(nType)[:, 1]
        nf = bg.nf[:, self.nTypeNum:]
        eId = bg.eId
        eType = bg.ef[:, :self.eTypeNum]
        nenType_id = self.get_nenType(nType, eType, eId)
        feature = bg.feature
        # graph===========================================
        x = self.gnn_block(nf, nId, nType, nType_id, eType, eId, nenType_id, training=training)
        # fc===========================================
        x = self.output_block(x, feature, training=training)
        return x

    def get_nenType(self, nType, eType, eId):
        nType_id = tf.where(nType)[:, 1]
        nType_id_s = tf.gather(nType_id, eId[0])
        nType_id_t = tf.gather(nType_id, eId[1])
        eType_id = tf.where(eType)[:, 1]
        nenType_id = tf.stack([nType_id_s, eType_id, nType_id_t])
        nenType_id = tf.transpose(nenType_id, [1, 0])
        return nenType_id

class Gnn_block(layers.Layer):
    def __init__(self, hiddenSize, nTypeNum, eTypeNum, graphSteps, heads=10, graphType=0, vntResidual=False):
        super(Gnn_block, self).__init__()
        self.hiddenSize = hiddenSize
        self.nTypeNum = nTypeNum
        self.eTypeNum = eTypeNum
        self.graphSteps = graphSteps
        self.preDense = layers.Dense(self.hiddenSize, activation='relu')
        self.preLn = layers.LayerNormalization(trainable=True)
        self.postDense = layers.Dense(self.hiddenSize, activation='relu')
        self.postLn = layers.LayerNormalization(trainable=True)
        if graphType == 0:
            self.heteroNode = True
            self.heteroEdge = True
            self.metaRelation = True
            self.updateType = 'gru'
            self.updateFunc = Update_gru(hiddenSize)
        elif graphType == 1:
            self.heteroNode = False
            self.heteroEdge = True
            self.metaRelation = True
            self.updateType = 'gru'
            self.updateFunc = Update_gru(hiddenSize)
        elif graphType == 2:
            self.heteroNode = True
            self.heteroEdge = False
            self.metaRelation = True
            self.updateType = 'gru'
            self.updateFunc = Update_gru(hiddenSize)
        elif graphType == 3:
            self.metaRelation = False
            self.heteroNode = True
            self.heteroEdge = True
            self.metaRelation = True
            self.updateType = 'gru'
            self.updateFunc = Update_gru(hiddenSize)
        elif graphType == 4:
            self.heteroNode = True
            self.heteroEdge = True
            self.metaRelation = True
            self.updateType = 'hgt'
            self.updateFunc = None
        else:
            raise Exception(f'graphType {graphType} is not supported')
        self.graphLayers = [GnnLayer_molHgt(hiddenSize, nTypeNum, eTypeNum, heads=heads,
                                            updateType=self.updateType,
                                            updateFunc=self.updateFunc,
                                            heteroNode=self.heteroNode,
                                            heteroEdge=self.heteroEdge,
                                            metaRelation=self.metaRelation)
                            for _ in range(self.graphSteps)]
        self.readouts = [Readout_vnt(self.hiddenSize, heads=heads, vntResidual=vntResidual)
                         for _ in range(self.graphSteps)]

    def call(self, nf, nId, nType, nType_id, eType, eId, nenType_id, training=False):
        nf_pre = self.preDense(nf)
        nf_pre = self.preLn(nf_pre, training=training)
        nf_i = nf_pre
        xs = []
        for i in range(self.graphSteps):
            nf_i = self.graphLayers[i](nf_i, nType, nType_id, eType, eId, nenType_id, training=training)
            x_i = self.readouts[i](nf_i, nId, training=training)
            xs.append(x_i)
        x = tf.concat(xs, axis=-1)
        x = self.postDense(x)
        x = self.postLn(x, training=training)
        return x
