import os
import time
import logging
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, Trials, hp, space_eval
from multiprocessing import Process, Lock
from . import tools_log
from . import tools_file
from . import tools_data_io

logging.getLogger('hyperopt.tpe').setLevel(logging.FATAL)
logging.getLogger('hyperopt.fmin').setLevel(logging.FATAL)
file_cache = 'hp_cache.pkl'
file_results = 'hp_results.csv'
file_progress = 'hp_progress.csv'
hpRepeat = 'hp repeat'
searchDone = 'search done'
lock = Lock()

def fmin_multiProcess(fminPs, sleepTime=1.0):
    if len(fminPs) > 1:
        ps = {}
        for i, fminP in enumerate(fminPs):
            p = Process(target=fminP.run)
            p.start()
            ps[i] = p
            time.sleep(sleepTime)
        for i in ps:
            ps[i].join()
    else:
        fminPs[0].run()
    print('')
    fminPs[0].saveResults()
    bestHp, id = fminPs[0].getBestHpAndId()
    return bestHp, id

class Fmin_process():

    def __init__(self, space, max_evals, timeout=None,
                 algo=tpe.suggest, rstate=None,
                 folder='temp', logger=None):
        self.space = space
        self.algo = algo
        self.max_evals = max_evals
        self.folder = folder
        tools_file.createFolder(self.folder)
        self.path_cache = os.path.join(folder, file_cache)
        self.path_results = os.path.join(folder, file_results)
        self.path_progress = os.path.join(folder, file_progress)
        self.io_process = tools_data_io.Lock_io_dict(self.path_progress)
        self.io_cache = tools_data_io.Lock_io_pkl(self.path_cache)
        self.rstate = rstate
        if logger is None:
            self.logger = tools_log.createLogger()
        else:
            self.logger = logger
        self.timeout = timeout

    def run(self):
        t0 = time.time()
        for i in range(self.max_evals * 2):
            t1 = time.time()
            if self.timeout > 0:
                t = (t1 - t0) / 3600.0
                if t >= self.timeout:
                    self.logger.info(f'\nt:{t:.3f}h >= {self.timeout:.3f}h, search done.')
                    break
            with lock:
                cache = self.io_cache.read()
                if cache is not None and 'trials' in cache:
                    trials = cache['trials']
                    if len(trials.trials) >= self.max_evals:
                        break
                else:
                    trials = Trials()
            # ==================================
            fmin_fail = False
            try:
                fmin(self.fn, self.space, algo=self.algo, max_evals=len(trials.trials) + 1,
                     trials=trials, rstate=self.rstate,
                     verbose=True, show_progressbar=False,
                     catch_eval_exceptions=False, return_argmin=False)
            except Exception as e:
                if str(e) == hpRepeat:
                    continue
                elif str(e) == searchDone:
                    break
                else:
                    self.logger.exception(e)
                    fmin_fail = True
            with lock:
                if len(trials.trials) > self.max_evals:
                    break
                if not fmin_fail and len(trials.trials) > 0:
                    cache = self.io_cache.read()
                    if cache is None:
                        cache = {}
                    if 'trials' in cache.keys():
                        trials_old = cache['trials']
                        trials_old.insert_trial_doc(trials.trials[-1])
                        trials_old.refresh()
                        trials = trials_old
                    cache['trials'] = trials
                    self.io_cache.write(cache)
                    t2 = time.time()
                    run_t = (t2 - t0) / 3600.0
                    eval_t = (t2 - t1) / 3600.0
                    log_dict = {
                        'evals': f'{len(trials.trials)}',
                        'eval_t': f'{eval_t:.3f}h',
                        'run_t': f'{run_t:.3f}h',
                        'loss': f'{trials.trials[-1]["result"]["loss"]:.4f}',
                        'best_loss': f'{trials.best_trial["result"]["loss"]:.4f}',
                    }
                    log = ", ".join([f'{k}:{log_dict[k]}' for k in log_dict])
                    self.io_process.write(log_dict, update=True)
                    self.logger.info(log, nextLine=True)

    def fn(self, hp):
        with lock:
            id, count, iState, iResult, = self.getState(hp)
            if count < self.max_evals:
                if id < 0:
                    self.updateState(hp, 1, None)
                    id, count, iState, iResult = self.getState(hp)
                else:
                    raise Exception(hpRepeat)
            else:
                raise Exception(searchDone)
        try:
            result = self.train(hp, id=id)
            with lock:
                self.updateState(hp, 2, result)
            return result
        except Exception as e:
            with lock:
                self.updateState(hp, 3, None)
            raise e

    def train(self, hp, id=None):
        pass

    def getState(self, iHp):
        cache = self.io_cache.read()
        count = 0
        if cache is not None and 'hp' in cache:
            count = len(cache['hp'])
            if iHp in cache['hp']:
                id = cache['hp'].index(iHp)
                return id, count, cache['hpState'][id], cache['hpResult'][id],
        return -1, count, None, None

    def updateState(self, iHp, iState, iResult):
        cache = self.io_cache.read()
        if cache is None:
            cache = {}
        if 'hp' in cache.keys():
            if iHp in cache['hp']:
                id = cache['hp'].index(iHp)
                cache['hpState'][id] = iState
                cache['hpResult'][id] = iResult
            else:
                cache['hp'].append(iHp)
                cache['hpState'].append(iState)
                cache['hpResult'].append(iResult)
        else:
            cache['hp'] = [iHp]
            cache['hpState'] = [iState]
            cache['hpResult'] = [iResult]
        self.io_cache.write(cache)

    def getBestHpAndId(self):
        cache = self.io_cache.read()
        if cache is not None:
            min = float('inf')
            id = -1
            for i in range(len(cache['hp'])):
                if cache['hpState'][i] == 2 and cache['hpResult'][i] < min:
                    min = cache['hpResult'][i]
                    id = i
            if id != -1:
                return cache['hp'][id], id
        return None, None

    def getHpFromTrials(self, trial):
        vals = trial["misc"]["vals"]
        rval = {}
        for k, v in list(vals.items()):
            if v:
                rval[k] = v[0]
        hp = space_eval(self.space, rval)
        return hp

    def saveResults(self):
        cache = self.io_cache.read()
        if cache is not None:
            datas = {k: [] for k in cache['hp'][0]}
            for i in range(len(cache['hp'])):
                for k in cache['hp'][i]:
                    datas[k].append(cache['hp'][i][k])
            datas['hpState'] = cache['hpState']
            datas['hpResult'] = cache['hpResult']
            df = pd.DataFrame(datas)
            df.to_csv(self.path_results, index=False)

class Test_train(Fmin_process):
    def __init__(self, evals=10, timeout=0.0, folder='ws', algo=tpe.suggest, rstate=None, ):
        space = {
            'p0': hp.quniform('p0', low=1, high=10000, q=1),
            'p1': hp.choice('p1', [i * 0.1 for i in range(10)]),
        }
        super().__init__(space, evals, timeout=timeout, algo=algo, rstate=rstate, folder=folder, logger=None)

    def train(self, hp, id=None):
        np.random.seed(int(hp['p0']))
        x = np.random.random()
        score = x + (hp['p1'] - 0.15) ** 2
        time.sleep(0.1)
        return score

def test():
    folder = 'ws/hp'
    tools_file.createFolder(folder, bDelFolder=True)
    ps = [Test_train(evals=100, timeout=0.0, folder=folder) for i in range(4)]
    best = fmin_multiProcess(ps, sleepTime=0.1)
    print('result', best)
