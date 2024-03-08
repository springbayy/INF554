import json
from pathlib import Path
import numpy as np
#import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#print(device)

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

path_to_training = Path("training")
path_to_test = Path("test")

#####
# training and test sets of transcription ids
#####
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])

#####
# naive_baseline: all utterances are predicted important (label 1)
#####
test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    test_labels[transcription_id] = [1] * len(transcription)

with open("test_labels_naive_baseline.json", "w") as file:
    json.dump(test_labels, file, indent=4)

#####
# text_baseline: utterances are embedded with SentenceTransformer, then train a classifier.
#####
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer

bert = SentenceTransformer('all-MiniLM-L6-v2', device="cuda:0")
print(bert.device)

speakerList=[]

y_training = []
with open("training_labels.json", "r") as file:
    training_labels = json.load(file)
X_training = []
for transcription_id in training_set:
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    for utterance in transcription:
        X_training.append(utterance["text"])
        speakerList.append(utterance["speaker"])
    
    y_training += training_labels[transcription_id]


n=len(X_training)
print(n)

trueList=[]
falseList=[]

for i in range(n):
    lenght=len(X_training[i])
    val=y_training[i]

    if val ==1:
        trueList.append(lenght)
    else: 
        falseList.append(lenght)

trueList=np.array(trueList)
falseList=np.array(falseList)

avgTrue=np.sum(trueList)/len(trueList)
avgFalse=np.sum(falseList)/len(falseList)

print('True, charachters: ', avgTrue)
print('False, charachters: ',avgFalse)

speakerUI=[]
speakerME=[]
speakerPM=[]
speakerID=[]



for i in range(n):
    speaker=speakerList[i]
    val=y_training[i]

    if speaker =="UI":
        speakerUI.append(val)
    elif speaker =="ME":
        speakerME.append(val)
    elif speaker =="PM":
        speakerPM.append(val)
    elif speaker =="ID":
        speakerID.append(val)


speakerID=np.array(speakerID)
speakerME=np.array(speakerME)
speakerPM=np.array(speakerPM)
speakerUI=np.array(speakerUI)

valID=np.sum(speakerID)/len(speakerID)
valME=np.sum(speakerME)/len(speakerME)
valPM=np.sum(speakerPM)/len(speakerPM)
valUI=np.sum(speakerUI)/len(speakerUI)

print("ID: ", valID)
print("ME: ", valME)
print("PM: ", valPM)
print("UI: ", valUI)