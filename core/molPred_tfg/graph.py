# --*-- coding: utf-8 --*--
# Copyright (c) 2019 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben
import os
import random
import threading
from queue import Queue
import tensorflow as tf
import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import HybridizationType
from ..toolPack import mol_format
from ..toolPack import tools_data
from ..toolPack import tools_data_io
from ..toolPack import mol_feature_ds
from ..toolPack import tools_log

class MolGraph():
    def __init__(self, pos=False, bFeature=False, batchSize=32, batchMax=None, logger=None):
        if logger is None:
            self.logger = tools_log.createLogger()
        else:
            self.logger = logger
        self.batchSize = batchSize
        if batchMax is None:
            self.batchType = 'graph'
            self.batchMax = float("inf")
        else:
            self.batchType = batchMax.split('_')[0]
            self.batchMax = int(batchMax.split('_')[1])
        self.pos = pos
        self.bFeature = bFeature
        self.ATOM_FEATURE = {
            'modelAtom': ['Other', 1, 6, 7, 8, 9, 15, 16, 17, 35, 53],
            'atom': list(range(1, 101)),
            'degree': ['Other', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'formalCharge': ['Other', -5, -4, -3, -2, -1, 0, 1, 2, 3],
            'numH': ['Other', 0, 1, 2, 3, 4, 5, 6],
            'hybridization': ['Other',
                              HybridizationType.S,
                              HybridizationType.SP,
                              HybridizationType.SP2,
                              HybridizationType.SP3,
                              HybridizationType.SP3D,
                              HybridizationType.SP3D2,
                              HybridizationType.OTHER],
            'aromatic': [''],
            'ring': ['Other', 0, 3, 4, 5, 6, 7, 8],
            'mass': ['']
        }

        self.BOND_FEATURE = {
            'bond': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'self'],
        }

        if self.pos:
            self.ATOM_FEATURE['p'] = ['x', 'y', 'z']
            self.BOND_FEATURE['d'] = ['x', 'y', 'z']

        self.atomFeature = {}
        for name in self.ATOM_FEATURE:
            for e in self.ATOM_FEATURE[name]:
                self.atomFeature[name + str(e)] = len(self.atomFeature)
        self.bondFeature = {}
        for name in self.BOND_FEATURE:
            for e in self.BOND_FEATURE[name]:
                self.bondFeature[name + str(e)] = len(self.bondFeature)
        self.atomFeatureNum = len(self.atomFeature)
        self.bondFeatureNum = len(self.bondFeature)

    def getGraph(self, data, bLog=False):
        try:
            smi_org = data[0]
            if len(data) > 1:
                yTrue = np.array(data[1:], dtype=np.float32)
                yMask = 1.0 - np.isnan(yTrue)
                yTrue[np.isnan(yTrue)] = 0.0
            else:
                yTrue = None
                yMask = None
            # =======================================================
            smi = mol_format.standardizeSmi(smi_org)
            if smi is None:
                if bLog:
                    print('standardizeSmi fail')
                return None
            # =======================================================
            if self.pos:
                mol = mol_format.smiToMol_optimizeConformer(smi)
            else:
                mol = mol_format.smiToMol(smi)
            # =======================================================
            if 'atom' in self.ATOM_FEATURE:
                mol = filter_atomType_mol(mol, bLog=bLog, atomTypes=self.ATOM_FEATURE['atom'])
            if mol is None:
                return None
            # =======================================================
            feature = self.getGraphFeature(smi)
            nf = self.getNodeFeature(mol)
            ef, eId = self.getEdgeFeature(mol)
            # ===============================================================
            graph = Graph(smi_org, nf, ef, eId, yTrue=yTrue, yMask=yMask, feature=feature)
            if (not self.batchType == 'graph') and self.getCountForBatchType(graph) >= self.batchMax:
                return None
            # ===============================================================
            return graph
        except Exception as e:
            if bLog:
                print(f'fail:{data[1]}')
            return None

    def getNodeFeature(self, mol):
        if self.pos:
            conf = mol.GetConformer()
        else:
            conf = None
        nf = []
        for i, atom in enumerate(mol.GetAtoms()):
            nf_i = np.zeros(self.atomFeatureNum, dtype=np.float32)
            # ---------------------------------
            if atom.GetAtomicNum() in self.ATOM_FEATURE['modelAtom']:
                nf_i[self.atomFeature['modelAtom' + str(atom.GetAtomicNum())]] = 1.0
            else:
                nf_i[self.atomFeature['modelAtomOther']] = 1.0
            # ---------------------------------
            if 'atom' in self.ATOM_FEATURE:
                nf_i[self.atomFeature['atom' + str(atom.GetAtomicNum())]] = 1.0
            # ---------------------------------
            if atom.GetTotalDegree() in self.ATOM_FEATURE['degree']:
                nf_i[self.atomFeature['degree' + str(atom.GetTotalDegree())]] = 1.0
            else:
                print('degree', atom.GetTotalDegree())
                nf_i[self.atomFeature['degreeOther']] = 1.0
            # ---------------------------------
            if atom.GetFormalCharge() in self.ATOM_FEATURE['formalCharge']:
                nf_i[self.atomFeature['formalCharge' + str(atom.GetFormalCharge())]] = 1.0
            else:
                print('formalCharge', atom.GetFormalCharge())
                nf_i[self.atomFeature['formalChargeOther']] = 1.0
            # ---------------------------------
            if atom.GetTotalNumHs() in self.ATOM_FEATURE['numH']:
                nf_i[self.atomFeature['numH' + str(atom.GetTotalNumHs())]] = 1.0
            else:
                print('numH', atom.GetTotalNumHs())
                nf_i[self.atomFeature['numHOther']] = 1.0
            # ---------------------------------
            if atom.GetHybridization() in self.ATOM_FEATURE['hybridization']:
                nf_i[self.atomFeature['hybridization' + str(atom.GetHybridization())]] = 1.0
            else:
                nf_i[self.atomFeature['hybridizationOther']] = 1.0
            # ---------------------------------
            if atom.GetIsAromatic():
                nf_i[self.atomFeature['aromatic']] = 1.0
            # ---------------------------------
            if atom.IsInRing():
                bOther = True
                for ringSize in self.ATOM_FEATURE['ring'][2:]:
                    if atom.IsInRingSize(ringSize):
                        nf_i[self.atomFeature['ring' + str(ringSize)]] = 1.0
                        bOther = False
                        break
                if bOther:
                    nf_i[self.atomFeature['ringOther']] = 1.0
            else:
                nf_i[self.atomFeature['ring0']] = 1.0
            # ---------------------------------
            nf_i[self.atomFeature['mass']] = atom.GetMass() * 0.01
            # ---------------------------------
            if self.pos:
                ps = np.array(conf.GetAtomPosition(i))
                nf_i[self.atomFeature['px']] = ps[0]
                nf_i[self.atomFeature['py']] = ps[1]
                nf_i[self.atomFeature['pz']] = ps[2]
            # ---------------------------------
            nf.append(nf_i)
        return np.array(nf)

    def getEdgeFeature(self, mol):
        atomNum = mol.GetNumAtoms()
        if self.pos:
            conf = mol.GetConformer()
        else:
            conf = None
        eId_i = []
        eId_j = []
        ef = []
        for i in range(atomNum):
            for j in range(atomNum):
                ef_ij = np.zeros(self.bondFeatureNum, dtype=np.float32)
                # ---------------------------------
                bond = None
                if i != j:
                    bond = mol.GetBondBetweenAtoms(i, j)
                    if bond is not None:
                        bt = bond.GetBondType()
                        ef_ij[[self.bondFeature['bond' + str(bt)]]] = 1.0
                else:
                    ef_ij[[self.bondFeature['bond' + 'self']]] = 1.0
                # ---------------------------------
                if self.pos:
                    d = np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j))
                    ef_ij[[self.bondFeature['dx']]] = d[0]
                    ef_ij[[self.bondFeature['dy']]] = d[1]
                    ef_ij[[self.bondFeature['dz']]] = d[2]
                # ---------------------------------
                if bond is not None or i == j:
                    eId_i.append(i)
                    eId_j.append(j)
                    ef.append(ef_ij)
        eId = [eId_j, eId_i]
        return np.array(ef), np.array(eId)

    def getGraphFeature(self, smi):
        if self.bFeature:
            feature = mol_feature_ds.getFeature_ds(smi)
            if feature is None:
                raise Exception(f'getGraphFeature fail:{smi}')
        else:
            feature = None
        return feature

    def getGraphs_fromFile(self, pathData, pathCache=None, predData=False, cpu=1):
        if pathCache is None:
            pathCache = os.path.splitext(pathData)[0] + '.graph'
        if os.path.exists(pathCache):
            graphs = tools_data_io.pkl_read(pathCache)
        else:
            df = pd.read_csv(pathData)
            if predData:
                df = df[['smiles']]
            datas = df.to_numpy()
            graphs = tools_data.multi(datas, self.getGraph, cpu=cpu, filterNone=True, bar=False)
            fail = len(datas) - len(graphs)
            self.logger.info(f'{pathData} getGraph fail:{fail}')
            tools_data_io.pkl_write(pathCache, graphs)
        return graphs

    def getCountForBatchType(self, graph):
        if self.batchType == 'node':
            count = graph.nf.shape[0]
        elif self.batchType == 'edge':
            count = graph.ef.shape[0]
        else:
            count = 0
        return count

    def toDataset(self, graphs, shuffle=False):

        if graphs[0].yTrue is not None:
            ys = [graph.yTrue for graph in graphs]
            yMean = np.mean(ys, axis=0)
            yStd = np.std(ys, axis=0)
        else:
            yMean = None
            yStd = None
            # ==================================================

        def datasetFunc():
            dataSize = len(graphs)
            ids = list(range(dataSize))
            if shuffle:
                random.shuffle(ids)
            batchGraphs = []
            batchMaxCount = 0
            graphCount = 0
            # ---------------------------------
            for id in ids:
                batchMaxCount = batchMaxCount + self.getCountForBatchType(graphs[id])
                if len(batchGraphs) < self.batchSize and batchMaxCount < self.batchMax:
                    batchGraphs.append(graphs[id])
                else:
                    graphCount += len(batchGraphs)
                    yield BatchGraph(batchGraphs)
                    batchGraphs = []
                    batchGraphs.append(graphs[id])
                    batchMaxCount = self.getCountForBatchType(graphs[id])
            # ---------------------------------
            if len(batchGraphs) > 0:
                graphCount += len(batchGraphs)
                yield BatchGraph(batchGraphs)
            # ---------------------------------
            if graphCount != dataSize:
                raise Exception('toDataset error,graphCount!=dataSize')

        return BufferGenerator(datasetFunc, bufferSize=100,
                               dataSize=len(graphs), yMean=yMean, yStd=yStd)

    def loadData(self, pathData, pathCache=None, predData=False, cpu=1, shuffle=False):
        graphs = self.getGraphs_fromFile(pathData, pathCache=pathCache, predData=predData, cpu=cpu)
        dataset = self.toDataset(graphs, shuffle=shuffle)
        return dataset

class Graph():

    def __init__(self, smi, nf, ef, eId, yTrue=None, yMask=None, feature=None):
        self.smi = smi
        self.nf = nf
        self.ef = ef
        self.eId = eId
        self.yTrue = yTrue
        self.yMask = yMask
        self.feature = feature

class BatchGraph():

    def __init__(self, graphs):
        self.graphs = graphs
        self.batchSize = len(self.graphs)
        self.toBatchGraph()

    def toBatchGraph(self):
        self.smi = []
        self.nf = []
        self.ef = []
        self.nId = []
        self.eId = []
        self.yTrue = []
        self.yMask = []
        self.feature = []
        sum = 0
        for i, graph in enumerate(self.graphs):
            self.smi.append(graph.smi)
            self.nf.append(graph.nf)
            self.ef.append(graph.ef)
            self.nId.append(tf.fill([len(graph.nf)], i))
            self.eId.append(graph.eId + sum)
            if graph.yTrue is not None:
                self.yTrue.append(graph.yTrue)
            if graph.yMask is not None:
                self.yMask.append(graph.yMask)
            if graph.feature is not None:
                self.feature.append(graph.feature)
            sum += len(graph.nf)
        self.smi = tf.cast(self.smi, tf.string)
        self.nId = tf.concat(self.nId, axis=0)
        self.nId = tf.cast(self.nId, tf.int32)
        self.nf = tf.concat(self.nf, axis=0)
        self.nf = tf.cast(self.nf, tf.float32)
        self.eId = tf.concat(self.eId, axis=1)
        self.eId = tf.cast(self.eId, tf.int32)
        self.ef = tf.concat(self.ef, axis=0)
        self.ef = tf.cast(self.ef, tf.float32)
        if len(self.yTrue) > 0:
            self.yTrue = tf.cast(self.yTrue, tf.float32)
        else:
            self.yTrue = None
        if len(self.yMask) > 0:
            self.yMask = tf.cast(self.yMask, tf.float32)
        else:
            self.yMask = None
        if len(self.feature) > 0:
            self.feature = tf.cast(self.feature, tf.float32)
        else:
            self.feature = None

class BufferGenerator():
    def __init__(self, generator, bufferSize=1, dataSize=None, yMean=None, yStd=None):
        self._generator = generator
        self.bufferSize = bufferSize
        self.dataSize = dataSize
        self.yMean = yMean
        self.yStd = yStd
        self.queue = Queue(maxsize=bufferSize)
        self.lock = threading.Lock()
        self.finish = False
        self.bStartBuffer = False

    def startBuffer(self):
        def buffer():
            with self.lock:
                self.finish = False
            for i in self._generator():
                self.queue.put(i)
            with self.lock:
                self.finish = True

        threading.Thread(target=buffer).start()

    def __call__(self):
        self.startBuffer()
        while (True):
            with self.lock:
                if self.finish and self.queue.qsize() == 0:
                    break
            yield self.queue.get()

def getTestBatchGraph(labels=1):
    datas = [['CCN'] + [0 for _ in range(labels)],
             ['C=C'] + [1 for _ in range(labels)]]
    molGraph = MolGraph(pos=False, bFeature=True)
    graphs = [molGraph.getGraph(data) for data in datas]
    batchGraph = BatchGraph(graphs)
    return batchGraph

def filter_atomType_mol(mol, atomTypes=[1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53], bLog=False):
    if mol is not None:
        for atom in mol.GetAtoms():
            at = atom.GetAtomicNum()
            if at not in atomTypes:
                if bLog:
                    print(f'atom type {at} is not in {atomTypes}')
                return None
        return mol
