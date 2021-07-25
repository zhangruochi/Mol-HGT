# coding=utf-8
# Copyright (c) 2020 Guangzhou fermion Technology Co.,Ltd. All rights reserved.
# create by Ben
import time
from .graph import *
from ..toolPack import tools_file
from ..toolPack import tools_data_io
from ..toolPack import tools_metric
from ..toolPack import tools_tf
from ..molPred.molPred import MolPred
from .tools_tfg import Model_ensemble

class MolPred_tfg(MolPred):
    def __init__(self, args, modelClass):
        super().__init__(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
        self.modelClass = modelClass
        # =======================================
        self.path_modelWeight = os.path.join(self.path_log, 'modelWeight.h5')
        self.path_args = os.path.join(self.path_log, 'modelArgs.bin')
        self.path_logTrain = os.path.join(self.path_log, 'train')
        self.path_logVal = os.path.join(self.path_log, 'val')
        # ======================================================
        self.molGraph = MolGraph(pos=False, bFeature=True,
                                 batchSize=self.args.batchSize,
                                 batchMax=self.args.batchMax,
                                 logger=None)
        if self.args.epochs == 0:
            if self.args.taskType == 'c':
                self.args.epochs = self.args.cEpochs
            else:
                self.args.epochs = self.args.rEpochs
        self.epochs = self.args.epochs * self.args.modelEnsemble

    def loadData_train(self):
        dataset_train = self.molGraph.loadData(self.path_train, pathCache=self.path_train_cache,
                                               predData=False, cpu=self.args.cpu, shuffle=True)
        dataset_val = self.molGraph.loadData(self.path_val, pathCache=self.path_val_cache,
                                             predData=False, cpu=self.args.cpu, shuffle=False)
        if os.path.exists(self.path_test):
            dataset_test = self.molGraph.loadData(self.path_test, pathCache=self.path_test_cache,
                                                  predData=False, cpu=self.args.cpu, shuffle=False)
            self.logger.info(f'{self.path_test} exists, eval on test')
        else:
            self.logger.info(f'{self.path_test} not exists, eval on val')
            dataset_test = dataset_val

        return (dataset_train, dataset_val, dataset_test)

    def __loadModel(self, training=True):
        if training:
            _args = self.args
        else:
            _args = tools_data_io.pkl_read(self.path_args)

        self.model = Model_ensemble(self.modelClass, _args)

        self.model(getTestBatchGraph(_args.labels), training=False)
        if not training:
            self.model.load_weights(self.path_modelWeight)

    def __saveModel(self):
        self.model.save_weights(self.path_modelWeight)
        tools_data_io.pkl_write(self.path_args, self.args)

    def _train(self):
        # with tf.device(f'/device:gpu:{self.args.gpu}'):
        return self.__train()

    def __train(self):
        tools_file.createFolder(self.path_logTrain)
        tools_file.createFolder(self.path_logVal)
        # data=======================================================
        self.dataset_train, self.dataset_val, self.dataset_test = self.loadData_train()
        self.args.yMean_train = self.dataset_train.yMean
        self.args.yStd_train = self.dataset_train.yStd
        # model=======================================================
        self.__loadModel(training=True)
        # writer=======================================================
        trainWriter = tf.summary.create_file_writer(self.path_logTrain)
        valWriter = tf.summary.create_file_writer(self.path_logVal)
        # lr=======================================================
        self.lr = tools_tf.LearningRate_epoch_batch(
            self.epochs, self.dataset_train.dataSize,
            warmupEpochs=self.args.warmupEpochs,
            lrStart=self.args.lrStart,
            lrMax=self.args.lrMax,
            lrEnd=self.args.lrEnd,
            curveType=self.args.lrCurveType,
            writer=trainWriter, name='lr')
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
        # metric=======================================================
        self.metric_train = tools_metric.Metric([self.args.metric])
        self.metric_val = tools_metric.Metric([self.args.metric])
        self.lossFn = tools_tf.getLossFunc(self.args)
        # train=======================================================
        failCount = 0
        monitorValue = 0
        for epoch in range(self.epochs):
            t0 = time.time()
            self.metric_train.resetState()
            self.metric_val.resetState()
            # train=======================================================
            self.train_epoch()
            with trainWriter.as_default():
                resultTrain = self.metric_train.epochCalc()
                for k in resultTrain:
                    tf.summary.scalar(f'epoch/{k}', resultTrain[k], step=epoch)
            # val=======================================================
            self.val_epoch()
            with valWriter.as_default():
                resultVal = self.metric_val.epochCalc()
                for k in resultVal:
                    tf.summary.scalar(f'epoch/{k}', resultVal[k], step=epoch)
            # log==================================
            t = time.time() - t0
            log = f'\r' \
                  f'epoch:{epoch:4d}, ' \
                  f't:{t:.2f}s, ' \
                  f'loss:{resultTrain["loss"]:.4f}, ' \
                  f'{self.args.metric}:{resultTrain[self.args.metric]:.4f}, ' \
                  f'val, ' \
                  f'loss:{resultVal["loss"]:.4f}, ' \
                  f'{self.args.metric}:{resultVal[self.args.metric]:.4f} '
            self.logger.info(log)
            # save===============================================
            if epoch == 0:
                monitorValue = resultVal[self.args.metric]
                self.__saveModel()
            else:
                monitorValue_epoch = resultVal[self.args.metric]
                if (tools_metric.metric_betters[self.args.metric] == 'max' and monitorValue_epoch >= monitorValue) \
                        or (tools_metric.metric_betters[self.args.metric] == 'min' and monitorValue_epoch <= monitorValue):
                    monitorValue = monitorValue_epoch
                    self.__saveModel()
                    failCount = 0
                else:
                    failCount += 1
                    if failCount >= self.args.earlyStop:
                        self.logger.info(f'{self.args.metric} failCount:{failCount},early stop')
                        break
        # =================================================================
        self.model = None
        result = self.__eval()
        return result

    def train_epoch(self):
        for bg in self.dataset_train():
            yTrue = bg.yTrue
            yMask = bg.yMask
            batchSize = bg.batchSize
            with tf.GradientTape() as tape:
                yPred = self.model(bg, training=True)
                loss = self.lossFn(yTrue, yPred, yMask=yMask)
            gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
            self.lr.updateBatchSize(batchSize)
            # ===============================================
            if not self.args.fileLog:
                results = self.metric_train.stepUpdate(yTrue.numpy(), yPred.numpy(), yMask.numpy(), loss=loss.numpy(), calc=True)
                log = f'batch:{bg.batchSize:4d}, ' \
                      f'loss:{results["loss"]:.4f}, ' \
                      f'{self.args.metric}:{results[self.args.metric]:.4f}'
                self.logger.info(log, nextLine=False)
            else:
                self.metric_train.stepUpdate(yTrue.numpy(), yPred.numpy(), yMask.numpy(), loss=loss.numpy(), calc=False)
        self.lr.updateEpoch()

    def val_epoch(self):
        for bg in self.dataset_val():
            yTrue = bg.yTrue
            yMask = bg.yMask
            yPred = self.model(bg, training=False)
            loss = self.lossFn(yTrue, yPred, yMask=yMask)
            self.metric_val.stepUpdate(yTrue.numpy(), yPred.numpy(), yMask.numpy(), loss=loss.numpy(), calc=False)

    def __eval(self):
        self.__loadModel(training=False)
        # ======================================
        yTrues = []
        yPreds = []
        yMasks = []
        for bg in self.dataset_test():
            yTrues.append(bg.yTrue.numpy())
            yPreds.append(self.model(bg, training=False).numpy())
            yMasks.append(bg.yMask.numpy())

        yTrue = np.concatenate(yTrues, axis=0)
        yPred = np.concatenate(yPreds, axis=0)
        yMask = np.concatenate(yMasks, axis=0)
        yMask = tools_metric.maskNan(yPred, yMask, logger=self.logger)
        result = {self.args.metric: tools_metric.metric_fns[self.args.metric](yTrue, yPred, yMask)}
        self.model = None
        return result

    def _pred(self):
        self.__loadModel(training=False)
        # ======================================
        dataset_pred = self.molGraph.loadData(self.path_predIn, pathCache=self.path_predIn_cache,
                                              predData=True,
                                              cpu=self.args.cpu, shuffle=False)
        # ======================================
        smis = []
        yPreds = []
        for bg in dataset_pred():
            smis.append(bg.smi)
            yPreds.append(self.model(bg, training=False))
        smis = tf.concat(smis, axis=0).numpy()
        yPreds = tf.concat(yPreds, axis=0).numpy()
        smis = [str(name, encoding="utf-8") for name in smis]
        self.model = None
        result = dict(zip(smis, yPreds))
        return result
