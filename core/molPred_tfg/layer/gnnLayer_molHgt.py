# --*-- coding: utf-8 --*--
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben 2020/3/30 下午1:36
from ..tools_tfg import *
from .update_gru import Update_gru
from .update_hgt import Update_hgt

class GnnLayer_molHgt(layers.Layer):

    def __init__(self, hiddenSize, nTypeNum, eTypeNum, heads=10,
                 updateType='gru', updateFunc=None,
                 heteroNode=True, heteroEdge=True, metaRelation=True, **kwargs):
        super(GnnLayer_molHgt, self).__init__(**kwargs)
        self.hiddenSize = hiddenSize
        self.nTypeNum = nTypeNum
        self.eTypeNum = eTypeNum
        self.heads = heads
        assert self.hiddenSize % heads == 0
        self.dk = int(self.hiddenSize / heads)
        self.aScale = self.dk ** 0.5
        self.heteroNode = heteroNode
        self.heteroEdge = heteroEdge
        self.metaRelation = metaRelation
        # ======================================-=====
        init_rn = tf.random_normal_initializer()
        init_one = tf.ones_initializer()
        init_zero = tf.zeros_initializer()
        # ===========================================
        if heteroNode:
            self.wq = tf.Variable(initial_value=init_zero([nTypeNum, hiddenSize, hiddenSize]), trainable=True, name=self.name + '_wq')
            self.bq = tf.Variable(initial_value=init_zero([1, nTypeNum, hiddenSize]), trainable=True, name=self.name + '_bq')
            # ----------------------------------------------
            self.wk = tf.Variable(initial_value=init_rn([nTypeNum, hiddenSize, hiddenSize]), trainable=True, name=self.name + '_wk')
            self.bk = tf.Variable(initial_value=init_zero([1, nTypeNum, hiddenSize]), trainable=True, name=self.name + '_bk')
            # ----------------------------------------------
            self.wv = tf.Variable(initial_value=init_rn([nTypeNum, hiddenSize, hiddenSize]), trainable=True, name=self.name + '_wv')
            self.bv = tf.Variable(initial_value=init_zero([1, nTypeNum, hiddenSize]), trainable=True, name=self.name + '_bv')
        else:
            self.denseQ = layers.Dense(self.hiddenSize)
            self.denseK = layers.Dense(self.hiddenSize)
            self.denseV = layers.Dense(self.hiddenSize)
        # ----------------------------------------------
        if heteroEdge:
            self.wEdge = tf.Variable(initial_value=init_rn([self.eTypeNum, self.heads, self.dk, self.dk]), trainable=True, name=self.name + '_wEdge')
            self.wMsg = tf.Variable(initial_value=init_rn([self.eTypeNum, self.heads, self.dk, self.dk]), trainable=True, name=self.name + '_wMsg')
        if metaRelation:
            self.relationScale = tf.Variable(initial_value=init_one([self.nTypeNum, self.eTypeNum, self.nTypeNum, self.heads]), trainable=True, name=self.name + '_relationScale')
        # ===========================================

        self.updateType = updateType
        if updateFunc is None:
            if self.updateType == 'gru':
                self.update = Update_gru(hiddenSize)
            elif self.updateType == 'hgt':
                self.update = Update_hgt(hiddenSize, nTypeNum)
            else:
                raise Exception(f'{self.updateType} is not supported')
        else:
            self.update = updateFunc
        self.act = gelu

    def call(self, nf, nType, nType_id, eType, eId, nenType_id, training=None):
        msg = self.getMessage(nf, nType, eType, eId, nenType_id, training=training)
        if self.updateType == 'gru':
            nf = self.update(nf, msg, training=training)
        elif self.updateType == 'hgt':
            nf = self.update(nf, nType, nType_id, msg, training=training)
        return nf

    def getMessage(self, nf, nType, eType, eId, nenType_id, training=None):
        id_s = eId[0]
        id_t = eId[1]
        # =============================================
        if self.heteroNode:
            nf_q = tf.einsum('ni,tio->nto', nf, self.wq) + self.bq
            nf_q = tf.einsum('nto,nt->no', nf_q, nType)
            nf_k = tf.einsum('ni,tio->nto', nf, self.wk) + self.bk
            nf_k = tf.einsum('nto,nt->no', nf_k, nType)
            nf_v = tf.einsum('ni,tio->nto', nf, self.wv) + self.bv
            nf_v = tf.einsum('nto,nt->no', nf_v, nType)
        else:
            nf_q = self.denseQ(nf)
            nf_k = self.denseK(nf)
            nf_v = self.denseV(nf)

        nf_q = tf.gather(nf_q, id_t)
        nf_k = tf.gather(nf_k, id_s)
        nf_v = tf.gather(nf_v, id_s)

        nf_q = tf.reshape(nf_q, [-1, self.heads, self.dk])
        nf_k = tf.reshape(nf_k, [-1, self.heads, self.dk])
        nf_v = tf.reshape(nf_v, [-1, self.heads, self.dk])
        # =============================================
        if self.heteroEdge:
            att = tf.einsum('nhi,thio->ntho', nf_q, self.wEdge)
            att = tf.einsum('nthi,nhi->nth', att, nf_k)
            att = tf.einsum('nth,nt->nh', att, eType)
        else:
            att = tf.reduce_sum(nf_q * nf_k, axis=2)
        if self.metaRelation:
            att = att * tf.gather_nd(self.relationScale, nenType_id)
        att = att / self.aScale
        att = segment_softmax(att, id_t)
        # =============================================
        if self.heteroEdge:
            msg = tf.einsum('nhi,thio->ntho', nf_v, self.wMsg)
            msg = tf.einsum('ntho,nt->nho', msg, eType)
        else:
            msg = nf_v * tf.expand_dims(att, axis=2)
        msg = msg * tf.expand_dims(att, axis=2)
        msg = tf.reshape(msg, [-1, self.hiddenSize])
        msg = tf.math.segment_sum(msg, id_t)
        msg = self.act(msg)
        return msg
