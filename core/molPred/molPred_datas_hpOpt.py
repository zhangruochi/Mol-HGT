# coding=utf-8
import argparse
from ..molPred.molPred_datas import *
from hyperopt import tpe
from ..toolPack.tools_hpOpt import fmin_multiProcess, Fmin_process, file_progress
from ..toolPack import tools_log
from ..toolPack.tools_metric import metric_betters
from ..toolPack import tools_data_io
from ..toolPack import tools_app

class HpTrain(Fmin_process):
    def __init__(self, args, gpu, folder, models):
        self.args = copy.deepcopy(args)
        self.args.gpu = gpu
        self.models = models
        space = self.models[self.args.model].getHpSpace()
        super(HpTrain, self).__init__(space, self.args.evals, timeout=self.args.timeout,
                                      algo=tpe.suggest, rstate=None, folder=folder, logger=None)

    def train(self, hp, id=None):
        args = HpTrain.getNewArgsFromHp(self.args, hp, id)
        iMp = MolPred_datas(args, self.models)
        scores, dfResult = iMp.train()
        iMp.close()
        score = scores[args.metric]
        if metric_betters[args.metric] == 'max':
            score = - score
        return score

    @classmethod
    def getNewArgsFromHp(clc, args, hp, id):
        args = copy.deepcopy(args)
        args_dict = vars(args)
        for k in hp:
            args_dict[k] = hp[k]
        args = argparse.Namespace(**args_dict)
        args.log = f'log_hp{id}'
        return args

class MolPred_datas_hpOpt:
    def __init__(self, args, models):
        self.args = copy.deepcopy(args)
        self.models = models
        self.path_hpFolder = os.path.join(self.args.dirOut, f'hp_{self.args.model}')
        self.path_hpBestResultFile = os.path.join(self.path_hpFolder, 'hp_bestResult.csv')
        self.path_hpBestArgs = os.path.join(self.path_hpFolder, 'hp_bestArgs.pkl')
        self.path_hpState = os.path.join(self.path_hpFolder, file_progress)
        self.bestArgs = None
        self.args.reTrain = 0

        if self.args.delHistory:
            clearLog(self.args)
            tools_file.delFolder(self.path_hpFolder)
        if self.args.fileLog:
            self.logger = tools_log.createLogger(name=self.path_hpFolder, folder=self.path_hpFolder, quiet=False)
        else:
            self.logger = tools_log.createLogger(quiet=False)

    def dataPreProcess(self):
        self.args.log = 'log_dataPreProcess'
        mp = MolPred_datas(self.args, self.models)
        mp.dataPreProcess()
        mp.close()

    def train(self):
        if not self.args.delHistory and os.path.exists(self.path_hpBestResultFile):
            self.logger.info(self.path_hpBestResultFile + ' exist')
            df_results = pd.read_csv(self.path_hpBestResultFile)
            return df_results
        # ==============================================
        if len(self.args.gpus) > 0:
            ps = [HpTrain(self.args, gpu, self.path_hpFolder, self.models) for gpu in self.args.gpus]
            bestHp, id = fmin_multiProcess(ps, sleepTime=1.0)
            if bestHp is not None:
                self.logger.info(bestHp)
                self.bestArgs = HpTrain.getNewArgsFromHp(self.args, bestHp, id)
                mp = MolPred_datas(self.bestArgs, self.models)
                scores, df_results = mp.train()
                mp.close()
                df_results.to_csv(self.path_hpBestResultFile, index=None)
                tools_data_io.pkl_write(self.path_hpBestArgs, self.bestArgs)
                if self.args.keepBestOnly:
                    self.clearLog_nonBest()
                return df_results
            else:
                self.logger.info('search fail')

    def clearLog_nonBest(self):
        if os.path.exists(self.path_hpBestArgs):
            self.bestArgs = tools_data_io.pkl_read(self.path_hpBestArgs)
            for iData in self.bestArgs.datas:
                iDataFolder = os.path.join(self.bestArgs.dirOut, iData, f'{self.bestArgs.model}')
                for folder in tools_file.getAllSubFolderPath(iDataFolder):
                    if not (self.bestArgs.log in folder):
                        tools_file.delFolder(folder)

    def pred(self):
        self.bestArgs = tools_data_io.pkl_read(self.path_hpBestArgs)
        mp = MolPred_datas(self.bestArgs, self.models)
        mp.pred()
        mp.close()

def addArgument_hpOpt(ap):
    ap.add_argument('--gpus', type=int, default=[0], nargs='+')
    ap.add_argument('--evals', type=int, default=20)
    ap.add_argument('--timeout', type=float, default=0.0)
    ap.add_argument('--delHistory', type=int, default=0)
    ap.add_argument('--keepBestOnly', type=int, default=1)

def main(models):
    option, ap = tools_app.optionParse('--model', None, choices=list(models.keys()))
    ap.add_argument('--dirIn', type=str, default='data/test_c')
    ap.add_argument('--dirOut', type=str, default='ws/test')
    addArgument_base(ap)
    addArgument_hpOpt(ap)
    models[option].addArgument_model(ap)
    # -------------------------------------------
    tools_app.run_bg()
    args = ap.parse_args()
    initArgs(args)
    tools_log.print_args(args)
    # -------------------------------------------
    args.fileLog = True
    # -------------------------------------------
    mp = MolPred_datas_hpOpt(args, models)
    io = tools_data_io.Lock_io_dict(mp.path_hpState)
    try:
        io.write({'state': 1}, update=True)
        mp.dataPreProcess()
        mp.train()
        mp.pred()
        io.write({'state': 2}, update=True)
    except Exception as e:
        mp.logger.exception(e)
        io.write({'state': 3}, update=True)
    finally:
        io.write({'state': 3}, update=True)
