import numpy as np
import json
from pathlib import Path
import scipy
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from progress.bar import Bar
import random
import torch
from ipywidgets import IntProgress
from IPython.display import display
from torchmetrics.classification import BinaryF1Score
from sklearn.metrics import f1_score
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

path_to_training = Path("training")
path_to_test = Path("test")

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]



def get_prevRelevant(set):

    set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in set])
    set.remove('IS1002a')
    set.remove('IS1005d')
    set.remove('TS3012c')

    for transcription_id in set:
        with open(path_to_training / f"{transcription_id}.txt", "r") as file:
            treeStructure = file.readlines()
    
        with open(path_to_training / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
    
    print(len(treeStructure))
            
    