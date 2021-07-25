# -*- coding: utf-8 -*-
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import logging
import copy
from . import tools_data

def init():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    RDLogger.logger().setLevel(val=4)

# ========================unique============================
def uniqueSmilesFromSelf(smis, log=False):
    length = len(smis)
    if length > 0:
        if type(smis) == list:
            smis1 = list(set(smis))
            smis1.sort(key=smis.index)
            length1 = len(smis1)
            repeat = length - length1
            ratio = repeat / length
            if log:
                logging.info('uniqueSmilesFromSelf, all:{}, remain:{}, repeat:{},repeatRatio:{:.3f}'.format(length, length1, repeat, ratio))
            return smis1
        elif type(smis) == dict:
            reverseSmiles = {v: k for k, v in smis.items()}
            smis1 = {v: k for k, v in reverseSmiles.items()}
            all = len(smis)
            remain = len(smis1)
            repeat = all - remain
            repeatRatio = repeat / all
            if log:
                logging.info('uniqueSmilesFromSelf, all:{}, remain:{}, repeat:{},repeatRatio:{:.3f}'.format(all, remain, repeat, repeatRatio))
            return smis1
        else:
            logging.warning('uniqueSmilesFromSelf,type error')
    else:
        logging.warning('uniqueSmilesFromSelf, fail,the size of smiles is 0')
    return None

def uniqueSmilesFromOthers_keepOrder(smiles, otherSmiles):
    length = len(smiles)
    with tqdm(otherSmiles) as bar:
        for e in bar:
            try:
                bar.set_description('uniqueSmilesFromOthers_keepOrder')
                smiles.remove(e)
            except:
                continue
    length1 = len(smiles)
    repeat = length - length1
    ratio = repeat / length
    print('uniqueSmilesFromSelf_keepOrder, all:{}, remain:{}, repeat:{},repeatRatio:{:.3f}'.format(length, length1, repeat, ratio))
    return smiles

def uniqueSmilesFromOthers_dict(smiles, otherSmiles):
    length = len(smiles)
    otherSmilesSet = set(otherSmiles.values())
    for k in list(smiles.keys()):
        if smiles[k] in otherSmilesSet:
            del smiles[k]
    length1 = len(smiles)
    print('uniqueSmilesFromOthers_dict,remain num:' + str(length1), 'remain ratio:' + str(length1 / length).zfill(3))
    return smiles

# ====================================================
def smiToMol(smi):
    if type(smi) == str and smi != '':
        mol = Chem.MolFromSmiles(smi)
        return mol

def smaToMol(sma):
    if type(sma) == str and sma != '':
        mol = Chem.MolFromSmarts(sma)
        return mol

def molToSmi(mol):
    if mol != None:
        smi = Chem.MolToSmiles(mol)
        if smi != '':
            return smi

def molToSma(mol):
    if mol != None:
        sma = Chem.MolToSmarts(mol)
        if sma != '':
            return sma

def inchiToSmi(inchi):
    smi = molToSmi(Chem.MolFromInchi(inchi))
    return smi

def inchiToSmis(inchis):
    smis = {}
    for k in inchis:
        try:
            smi = molToSmi(Chem.MolFromInchi(inchis[k]))
            if smi is not None:
                smis[k] = smi
        except:
            continue
    fail = len(inchis) - len(smis)
    if fail > 0:
        print('inchisToSmiles_dict,all:{},remain:{},fail:{},failRatio:{:.3f}'.format(len(inchis), len(smis), fail, fail / len(inchis)))
    return smis

def standardizeSma(sma):
    mol = smaToMol(sma)
    sma = molToSma(mol)
    return sma

def standardizeSmi(smi, kekuleSmiles=False):
    smi = molToSmi(smiToMol(smi))
    if kekuleSmiles:
        mol = smiToMol(smi)
        if mol:
            Chem.Kekulize(mol)
            smi_keku = Chem.MolToSmiles(mol, kekuleSmiles=True)
            if smi_keku != '':
                return smi_keku
    else:
        return smi

def smiToMols(smis, cpu=1, filterNone=True, bar=False):
    return tools_data.multi(smis, smiToMol, cpu=cpu, filterNone=filterNone, bar=bar)

def smaToMols(smas, cpu=1, filterNone=True, bar=False):
    return tools_data.multi(smas, smaToMol, cpu=cpu, filterNone=filterNone, bar=bar)

def molToSmis(mols, cpu=1, filterNone=True, bar=False):
    return tools_data.multi(mols, molToSmi, cpu=cpu, filterNone=filterNone, bar=bar)

def molToSmas(mols, cpu=1, filterNone=True, bar=False):
    return tools_data.multi(mols, molToSma, cpu=cpu, filterNone=filterNone, bar=bar)

def standardizeSmis(smis, kekuleSmiles=False, cpu=1, filterNone=True, bar=False):
    return tools_data.multi(smis, standardizeSmi, cpu=cpu, filterNone=filterNone, bar=bar, kekuleSmiles=kekuleSmiles)

# ====================================================
def smiToMol_optimizeConformer(smi, maxIters=400):
    mols = smiToMol_optimizeConformers(smi, numConfs=1, maxIters=maxIters)
    if mols is not None and len(mols) == 1:
        return mols[0]

def smiToMol_optimizeConformers(smi, numConfs=1, maxIters=400):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)
            mols = []
            for i in range(numConfs):
                iMol = copy.deepcopy(mol)
                AllChem.EmbedMolecule(iMol, randomSeed=i)
                AllChem.MMFFOptimizeMolecule(iMol, maxIters=maxIters)
                mols.append(iMol)
            return mols
    except:
        return None
