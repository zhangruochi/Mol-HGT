import os
import pandas as pd
import numpy as np
import copy
import argparse
from ..toolPack import tools_file
from ..toolPack import tools_log
from ..toolPack import tools_data_io

class MolPred():
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.dataName = os.path.basename(self.args.dirIn)
        self.path_dataIn = os.path.join(self.args.dirIn, self.args.data)
        self.path_dataOut = os.path.join(self.args.dirOut, self.args.data, self.args.model)
        self.path_log = os.path.join(self.path_dataOut, self.args.log)
        tools_file.createFolder(self.path_log, bDelFolder=False)
        if self.args.fileLog:
            self.logger = tools_log.createLogger(name=self.path_log, folder=self.path_log, quiet=False)
        else:
            self.logger = tools_log.createLogger(quiet=False)
        # =====================================================
        self.path_train = os.path.join(self.path_dataIn, 'train.csv')
        self.path_val = os.path.join(self.path_dataIn, 'val.csv')
        self.path_test = os.path.join(self.path_dataIn, 'test.csv')
        self.path_predIn = os.path.join(self.path_dataIn, 'pred.csv')
        # =====================================================
        self.path_train_cache = os.path.join(self.path_dataOut, 'train.bin')
        self.path_val_cache = os.path.join(self.path_dataOut, 'val.bin')
        self.path_test_cache = os.path.join(self.path_dataOut, 'test.bin')
        self.path_predIn_cache = os.path.join(self.path_dataOut, 'pred.bin')
        # =====================================================
        self.path_eval = os.path.join(self.path_log, 'eval.csv')
        self.path_predOut = os.path.join(self.path_log, 'pred.csv')
        self.path_config = os.path.join(self.path_log, 'config.json')
        # =====================================================
        if os.path.exists(self.path_train):
            if self.args.labels == 0:
                self.args.labels = pd.read_csv(self.path_train, nrows=1).shape[1] - 1
        else:
            raise Exception(f'trainData:{self.path_train} do not exist.')

    def loadData_path(self, pathData, pathCache, predData=False):
        return None

    def loadData_train(self):
        train_set = self.loadData_path(self.path_train, self.path_train_cache, predData=False)
        val_set = self.loadData_path(self.path_val, self.path_val_cache, predData=False)
        if os.path.exists(self.path_test):
            test_set = self.loadData_path(self.path_test, self.path_test_cache, predData=False)
            self.logger.info(f'{self.path_test} exists, eval on test')
        else:
            self.logger.info(f'{self.path_test} not exists, eval on val')
            test_set = val_set

        return train_set, val_set, test_set

    def dataPreprocess(self):
        self.loadData_train()

    def writeConfig(self):
        args_dict = vars(self.args)
        for k in self.args.config:
            if self.args.config[k] is None:
                args_dict[k] = None
            else:
                args_dict[k] = type(self.args.config[k])(args_dict[k])
            self.args.config[k] = args_dict[k]
        tools_data_io.json_write(self.path_config, self.args.config)

    def readConfig(self):
        self.args.config = tools_data_io.json_read(self.path_config)
        args_dict = vars(self.args)
        for k in self.args.config:
            args_dict[k] = self.args.config[k]
        self.args = argparse.Namespace(**args_dict)

    def train(self):
        if not self.args.reTrain and os.path.exists(self.path_eval):
            self.logger.info(f'{self.path_eval} exists')
            df_eval = pd.read_csv(self.path_eval)
            return df_eval
        else:
            self.writeConfig()
            _result = self._train()
            result = {'dataName': self.dataName, 'data': self.args.data, 'model': self.args.model}
            result.update(_result)
            result.update(self.args.config)
            df_eval = pd.DataFrame({k: [v] for k, v in result.items()})
            df_eval.to_csv(self.path_eval, index=False)
            self.logger.info('train finish.\n')
            return df_eval

    def _train(self):
        return None

    def pred(self):
        if os.path.exists(self.path_predIn):
            self.readConfig()
            result = self._pred()
            df = pd.read_csv(self.path_predIn)
            smis = df['smiles'].to_list()
            preds = []
            for smi in smis:
                if smi in result:
                    preds.append(result[smi])
                else:
                    self.logger.info(f'{smi} pred fail.')
                    preds.append([None for _ in range(self.args.labels)])
            preds = np.array(preds)
            if self.args.labels == 1:
                df[f'{self.args.model}_pred'] = preds[:, 0]
            else:
                for i in range(self.args.labels):
                    df[f'{self.args.model}_pred{i}'] = preds[:, i]
            df.to_csv(self.path_predOut, index=False)
            self.logger.info('pred finish.')

    def _pred(self):
        pass

    def close(self):
        self.logger.shutdown()
