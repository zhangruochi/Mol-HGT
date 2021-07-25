import os
import copy
import pandas as pd
from ..toolPack import tools_file
from ..toolPack import tools_app
from ..toolPack import tools_log

def initArgs(args):
    if args.taskType == None:
        if args.dirIn.endswith('_c'):
            args.taskType = 'c'
        elif args.dirIn.endswith('_r'):
            args.taskType = 'r'
        else:
            raise Exception(f'{args.dirIn} not endswith _c or _r.')
    if args.taskType == 'c':
        args.metric = args.cMetric
    else:
        args.metric = args.rMetric
    return args

def addArgument_base(ap):
    ap.add_argument('--log', type=str, default='log')
    ap.add_argument('--datas', type=str, default=['data'], nargs='+')
    ap.add_argument('--labels', type=int, default=0)
    ap.add_argument('--taskType', type=str, default=None)
    ap.add_argument('--cMetric', type=str, default='roc_auc', choices=['roc_auc', 'prc_auc'])
    ap.add_argument('--rMetric', type=str, default='rmse', choices=['r2', 'mae', 'rmse'])
    ap.add_argument('--cpu', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--reTrain', type=int, default=0)

def clearLog(args):
    for iData in args.datas:
        iPath = os.path.join(args.dirOut, iData, args.model)
        tools_file.clearSubFolders(iPath)

class MolPred_datas():
    def __init__(self, args, models):
        self.args = copy.deepcopy(args)
        self.mps = []
        for iData in self.args.datas:
            iArgs = copy.deepcopy(self.args)
            iArgs.data = iData
            iArgs.config = models[self.args.model].config_default
            self.mps.append(models[self.args.model].getMolPred(iArgs))

    def dataPreProcess(self):
        for iMp in self.mps:
            iMp.dataPreprocess()

    def train(self):
        dfs = []
        for iMp in self.mps:
            dfs.append(iMp.train())
        df_all = pd.concat(dfs, axis=0)
        df_stats = df_all.iloc[:, 3:4]
        mean = dict(df_stats.mean())
        std = dict(df_stats.std())
        df_result = pd.DataFrame(df_all.iloc[0:1], copy=True)
        for k in mean:
            if len(dfs) > 1:
                k_mean_std = f'{mean[k]:.4f}±{std[k]:.4f}'
            else:
                k_mean_std = f'{mean[k]:.4f}'
            df_result[k] = [k_mean_std]
        df_result['data'] = ['_'.join(df_all['data'].to_list())]
        if not self.args.fileLog:
            if len(dfs) > 1:
                print(pd.concat([df_all, df_result], axis=0).iloc[:, 0:4])
            else:
                print(df_result.iloc[:, 0:4])
        return mean, df_result

    def pred(self):
        for iMp in self.mps:
            iMp.pred()

    def close(self):
        for iMp in self.mps:
            iMp.close()

def main(models):
    option, ap = tools_app.optionParse('--model', None, choices=list(models.keys()))
    ap.add_argument('--dirIn', type=str, default='data/test_c')
    ap.add_argument('--dirOut', type=str, default='ws/test')
    addArgument_base(ap)
    models[option].addArgument_model(ap)
    # ============================================================
    tools_app.run_bg()
    args = ap.parse_args()
    tools_log.print_args(args)
    initArgs(args)
    # ============================================================
    args.fileLog = False
    mp = MolPred_datas(args,models)
    mp.train()
    mp.pred()
