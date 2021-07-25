# MolHGT

MolHGT: a heterogeneous graph neural network for predicting the properties of chemical molecules

## Install

### Packages
```
1. tensorflow
2. hyperopt
3. rdkit
4. matplotlib
5. scikit-learn
6. tqdm

Specific reference:
molHgt.yaml
```

## Content
* **data**  
The molnet benchmark data set is divided into train/val/test according to different random seeds.
* **core**  
Main code
* **descriptastorus**  
External feature calculation code, from https://github.com/bp-kelley/descriptastorus

## Training script
* **main_model.py**
It can be trained and evaluated. The data set is divided into tran/val/test dataset with multiple random seeds, and the evaluation result is in eval.csv.
* **main_model_hpOpt.py**:
You can search hyperparameters for training and evaluation. The data set is divided into tran/val/test dataset with multiple random seeds, and the evaluation result file is hp_bestResult.csv.
* **main_model_hpOpt_datas.py**
You can search hyperparameters for training and evaluation. The data set is divided into train/val/test with multiple datasets and multiple random seeds. The evaluation result are in model_hpOpt_datas_result_c.csv and model_hpOpt_datas_result_r.csv.

## Parameters
* **model**: Model type, must be molHgt.
* **dirIn**: Data input path.
* **dirOut**: The result output path.
* **datas**: The name of the data folder, the default is data.
* **log**: The name of the log folder, the default is log. Automatically named when using hyperparameter search.
* **cpu**: The number of data preprocessing cpu cores. The default is 0. When the value is 0, all cpu cores are used.
* **labels**: The number of labels in the data set, the default is 0. When the value is 0, the number of labels in the data set is automatically extracted.
* **taskType**: task type, c: classification, r: regression, the default is None. When the value is None, judge whether it is a classification or regression task at the end of the data folder noun.
* **epochs**: The number of training epochs. When the value is 0, select rEpochs or cEpochs according to the task type.
* **cEpochs**: The number of training epochs for the default classification task.
* **rEpochs**: The number of training epochs for the default regression task.
* **earlyStop**: The number of epochs for early stop training, the default is 10.
* **extFeat**: external feature, whether to use external features. 0: not used, 1: used.
* **graphType**: Graph network type, used in the ablation experiment of the paper. 0: Standard model, 1: Remove heterogeneous points, 2: Remove heterogeneous edges, 3: Remove metaRelation, 4: Use the default update function of the HGT paper.
* **hiddenSize**: The hidden layer dimension of the graph network, the number is an integer multiple of the heads. Automatically set when using hyperparameter search.
* **graphSteps**: The number of graph network layers. Automatically set when using hyperparameter search.
* **heads**: The number of heads of multiHead attention. The default is 10
* **lrMax**: Maximum learning rate. The default is 1e-4.
* **modelEnsemble**: The number of model integrations. The default is 1.
* **fcEnsemble**: fc layer integration quantity. The default is 1.
* **gpus**: The gpu id that can be used when using hyperparameter search.
* **evals**: The number of hyperparameter search trainings, the default is 20.
* **delHistory**: Used in hyperparameter search, whether to delete all old training logs (including model and evaluation results). 0: Do not delete, 1: Delete. The default is 0.
* **keepBestOnly**: Whether to keep only the best training log (including model and evaluation results) when using hyperparameter search. 0: not reserved, 1: reserved. The default is 1.
* **bg/bgt**: bg: run in the background without hanging up. bgt: The background does not hang up and the log is automatically displayed.

## Script for training model

```shell
# Run the hyperparameter search training of 10 data sets of molnet at one time.
# Use the standard model, use external features, use 4 GPUs (id 0 1 2 3), use 3 seed data, and train each seed 20 times in each data set.
python main_model_hpOpt_datas.py --model molHgt --dirIn data/molnet --dirOut ws --extFeat 1 --gpus 0 1 2 3 --datas seed0 seed1 seed2 --evals 20 --bgt ws/log
```
