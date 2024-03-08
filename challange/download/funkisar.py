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
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print('Device:', device)
#torch.set_default_tensor_type('torch.cuda.FloatTensor')


path_to_training = Path("training")
path_to_test = Path("test")

def get_grad(y_pred, y_true):
    beta=1
    beta2=beta**2
    grad=(((1+beta2)*y_true)/np.sum(y_pred+(beta2*y_true)))-((1+beta2)*np.dot(y_pred, y_true)/(np.sum(y_pred+(beta2*y_true))**2))

    return -grad

def mse_loss(y_pred, y_val):
    # l(y_val, y_pred) = (y_val-y_pred)**2
    grad = 2*(y_val-y_pred)
    hess = np.repeat(2,y_val.shape[0])
    return grad, hess 

        
def get_convoLines (transcription_id):

    #input: filename
    #output: list of lists [[],[],[]] of indicies of each subconversatin AND matrix of various features ANS list of sentence for each node

    with open(path_to_training / f"{transcription_id}.txt", "r") as file:
        treeStructure = file.readlines()
    
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)

    n=int((treeStructure[-1].split())[2])
    matrix=np.zeros((n+1,n+1))
    nChildrenList=[0]*(n+1)
    isDeviationList=[0]*(n+1)
    branchesList=[]
    actionToNodeList=[0]*(n+1)
    speakerList=[0]*(n+1)
    speakerToList=[0]*(n+1)
    nCharacterList=[0]*(n+1)
    sentenceList=['']*(n+1)
    nWordList=[0]*(n+1)
    haveVocalSound=[0]*(n+1)
    havedisfmarker=[0]*(n+1)
    haveQuestion=[0]*(n+1)
    hasGap=[0]*(n+1)

    mapSpeakerToInteger={
        'PM':1,
        'ME':2,
        'ID':3,
        'UI':4
    }

    mapActionToInteger={
        'Continuation': 1,
        'Explanation':2,
        'Elaboration':3,
        'Acknowledgement':4,
        'Result': 5,
        'Question-answer_pair':6,
        'Clarification_question':7,
        'Contrast':8,
        'Background':9,
        'Narration':10,
        'Alternation':11,
        'Q-Elab':12,
        'Conditional':13,
        'Correction':14,
        'Parallel':15,
        'Comment': 16
    }

    for utterance in (transcription):
        speaker=utterance["speaker"]
        sentence=utterance["text"]
        index=utterance["index"]
        nWord=len(sentence.split())
        nCharacters=len(sentence)

        if '?' in sentence:
            haveQuestion[index]=1

        if '<vocalsound>' in sentence:
            haveVocalSound[index]=1
        elif '<disfmarker>' in sentence:
            havedisfmarker[index]=1
        elif '<gap>' in sentence:
            havedisfmarker[index]=1

        sentence=sentence.replace('<vocalsound> ', '')
        sentence=sentence.replace(' <vocalsound>', '')
        sentence=sentence.replace('<vocalsound>', '')

        sentence=sentence.replace('<disfmarker> ', '')
        sentence=sentence.replace(' <disfmarker>', '')
        sentence=sentence.replace('<disfmarker>', '')

        sentence=sentence.replace('<gap> ', '')
        sentence=sentence.replace(' <gap>', '')
        sentence=sentence.replace('<gap>', '')

        sentenceList[index]=sentence
        speakerList[index]=mapSpeakerToInteger[speaker]
        nCharacterList[index]=nCharacters
        nWordList[index]=nWord
    

    for line in reversed(treeStructure):
        linelist=line.split()
        fromNode=int(linelist[0])
        toNode=int(linelist[2])
        matrix[toNode, fromNode]=1

        action=linelist[1]  
        actionToNodeList[toNode]=mapActionToInteger[action]

    endpointList=[]    
    for index in range(n+1):
        if np.sum(matrix[:, index])==0:
            endpointList.append(index)


    for endpoint in endpointList:
        convo=[]
        index=endpoint
        while index > 0:
            convo.append(index)
            search=matrix[index,:]
            indexplaceholder=np.where(search==1)
            indexplaceholder=indexplaceholder[0][0]

            speakerToList[index]=speakerList[indexplaceholder]
            index=indexplaceholder
            
        convo.append(0)
        branchesList.append(convo[::-1])

    mainConvo=branchesList[-1]
    for index in range(n+1):
        #number of children
        nChildrenList[index]=int(np.sum(matrix[:, index]))

        #is a side-branch
        if index not in mainConvo:
            isDeviationList[index]=1
    
    

    isDeviationList=np.array(isDeviationList)
    nCharacterList=np.array(nChildrenList)
    actionToNodeList=np.array(actionToNodeList)
    nWordList=np.array(nWordList)
    nCharacterList=np.array(nCharacterList)
    speakerToList=np.array(speakerToList)
    speakerList=np.array(speakerList)
    haveVocalSound=np.array(haveVocalSound)
    havedisfmarker=np.array(havedisfmarker)
    haveVocalSound=np.array(haveQuestion)
    havedisfmarker=np.array(hasGap)
    haveQuestion=np.array(haveQuestion)

    featureHash={
        'isDeviationList':isDeviationList,
        #'nCharacterList':nCharacterList,
        'actionToNodeList':actionToNodeList,
        #'haveVocalSound':haveVocalSound,
        'havedisfmarker':havedisfmarker,
        'haveVocalSound':haveVocalSound,
        'haveQuestion':haveQuestion,
        #'speakerList':speakerList,
    }

    featureMatrix=np.c_[isDeviationList, nChildrenList, hasGap, haveQuestion, actionToNodeList, nWordList, nCharacterList, haveVocalSound, havedisfmarker]

    sentenceList=np.array(sentenceList)
    
    return branchesList, featureHash, sentenceList, matrix

def get_convolineList(set):

    convolineList=[]
    try:
        set=np.random.choice(set, nTrees, replace=False)   
    except:
        pass


    for transtriction_id in set:
        branchesList, featureMatrix, sentenceList, matrix = get_convoLines(transtriction_id)

        try:
            branchesList=random.sample(branchesList, nSample)
        except:
            pass

        convolineList.append([branchesList, featureMatrix, sentenceList, transtriction_id, matrix])
    
    return(convolineList)

def get_matrixForTree(transcription_id, bert):

    convoline=get_convoLines(transcription_id)
    branchesList=convoline[0]
    featureHash=convoline[1]
    sentenceList=convoline[2]


    X_training_forTree=bert.encode(sentenceList)
    #X_training_forTree=torch.from_numpy(X_training_forTree).to(device)

    prevConvoList=[]
    n=len(X_training_forTree[:,0])
    for index in range(n):
        for branch in branchesList:
            if index in branch:
                break

        listIndex=branch.index(index)

        prevConvoIndexList=branch[max(listIndex-10, 0): listIndex+1]
        prevConvoIndexList.pop()
        prevSentence=''
        for prevIndex in prevConvoIndexList:
            prevSentence+=sentenceList[prevIndex]+ ' '
        
        prevConvoList.append(prevSentence)
    
    prevconvoEncoded=bert.encode(prevConvoList)
    X_training_forTree=np.append(X_training_forTree, prevconvoEncoded, axis=1)

    for feature in featureHash:
        data=featureHash[feature].reshape(-1,1)

        X_training_forTree=np.append(X_training_forTree, data, axis=1)


    
    return X_training_forTree


def get_MatrixFromSet(set):

    set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in set])

    try:
        set.remove('IS1002a')
    except:
        pass
    try:
        set.remove('IS1005d')
    except:
        pass
    try:
        set.remove('TS3012c')
    except:
        pass


    bert = SentenceTransformer('all-mpnet-base-v2')
    print('bert device: ', bert.device)


    with open("training_labels.json", "r") as file:
        training_labels = json.load(file)
        

    l=len(set)
    pbar = tqdm(total=l, desc="Encoding")
    for transcription_id in set:
        y=training_labels[transcription_id]
        X_matrix=get_matrixForTree(transcription_id, bert)

        try:
            X_training=np.append(X_training, X_matrix, axis=0)
            y_training=np.append(y_training, y)
        except:
            y_training=y
            X_training=X_matrix
        
        pbar.update(1)
    pbar.close()
                     
    return(X_training, y_training)

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

def get_modelFromSet(training_set):


    bst = XGBClassifier(tree_method="hist", device="cuda:0")
    print('bst device: ', bst.device)

    X_training, y_training = get_MatrixFromSet(training_set)

    print('Fitting Data')
    bst.fit(X_training, y_training)

    return bst

def get_preds(model, X_test):

    preds=model.predict(X_test)

    return preds


def get_labels(set):

    with open("training_labels.json", "r") as file:
        training_labels = json.load(file)
    
    set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in set])

    try:
        set.remove('IS1002a')
    except:
        pass
    try:
        set.remove('IS1005d')
    except:
        pass
    try:
        set.remove('TS3012c')
    except:
        pass

    y_labels=[]
    for tree in set:
        y_labels.append(training_labels[tree])

def get_scoreFromset(model, set):
    
    truematrix, y_labels=get_MatrixFromSet(set)
    preds=get_preds(model, truematrix)

    score=f1_score(y_labels, preds)
    return score




