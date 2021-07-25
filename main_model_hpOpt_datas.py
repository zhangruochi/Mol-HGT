# --*-- coding: utf-8 --*--
import core.molPred.init
from models import models
from core.molPred.molPred_datas_hpOpt import *
logger = tools_log.createLogger()

def hpTrain_datas(args, tasks=None):

    pathResult_c = os.path.join(args.dirOut, f'model_hpOpt_datas_result_c.csv')
    pathResult_r = os.path.join(args.dirOut, f'model_hpOpt_datas_result_r.csv')
    tools_file.delFile(pathResult_c)
    tools_file.delFile(pathResult_r)
    dfs_c = []
    dfs_r = []
    if tasks is None:
        tasks = tools_file.getAllSubFolderName(args.dirIn)
    for name in tasks:
        iArgs = copy.deepcopy(args)
        iArgs.dirIn = os.path.join(args.dirIn, name)
        iArgs.dirOut = os.path.join(args.dirOut, name)
        initArgs(iArgs)
        logger.info(f'hp search: {name} ')
        iMp = MolPred_datas_hpOpt(iArgs, models)
        logger.info('preprocessing...')
        iMp.dataPreProcess()
        logger.info('training...')
        iDf_result = iMp.train()
        if iDf_result is not None:
            iDf_result = pd.DataFrame(iDf_result.iloc[-1:, :4])
            if iArgs.taskType == 'c':
                dfs_c.append(iDf_result)
                df_c = pd.concat(dfs_c, axis=0)
                df_c.to_csv(pathResult_c, index=False)
            else:
                dfs_r.append(iDf_result)
                df_r = pd.concat(dfs_r, axis=0)
                df_r.to_csv(pathResult_r, index=False)
        logger.info('=' * 50)

if __name__ == '__main__':
    option, ap = tools_app.optionParse('--model', None, choices=list(models.keys()))
    ap.add_argument('--dirIn', type=str, default='data/molnet')
    ap.add_argument('--dirOut', type=str, default='ws')
    addArgument_base(ap)
    addArgument_hpOpt(ap)
    models[option].addArgument_model(ap)
    # -------------------------------------------
    tools_app.run_bg()
    args = ap.parse_args()
    tools_log.print_args(args,logger=logger)
    args.fileLog = True
    # ====================================
    tasks = [
        'molnet_random_delaney_ESOL_r',
        'molnet_random_sampl_freesolv_r',
        'molnet_random_lipo_r',
        # -----------------------------------
        'molnet_scaffold_hiv_c',
        'molnet_scaffold_bace_c',
        'molnet_scaffold_bbbp_c',
        # ----------------------------------
        'molnet_random_tox21_c',
        'molnet_random_toxcast_c',
        'molnet_random_sider_c',
        'molnet_random_clintox_c',
    ]
    hpTrain_datas(args, tasks=tasks)
