import os, logging
from rdkit import RDLogger

RDLogger.logger().setLevel(val=4)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
