from funkisar import*
from torch_geometric.data import Data
import torch_geometric
import scipy
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Predictor:
    def __init__(self) -> None:
        pass

    def get_convoLines (transcription_id):

        #input: filename
        #output: list of lists [[],[],[]] of indicies of each subconversatin AND matrix of various features ANS list of sentence for each node

        with open(path_to_training / f"{transcription_id}.txt", "r") as file:
            treeStructure = file.readlines()
        
        with open(path_to_training / f"{transcription_id}.json", "r") as file:
            transcription = json.load(file)
        
        with open("training_labels.json", "r") as file:
            training_labels = json.load(file)

        y_training=np.array(training_labels[transcription_id])

        y_training=torch.tensor(y_training)

        n=int((treeStructure[-1].split())[2])
        matrix=torch.zeros((n+1,n+1))
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
            if '<vocalsound>' in sentence:
                haveVocalSound[index]=1
            elif '<disfmarker>' in sentence:
                havedisfmarker[index]=1

            sentence=sentence.replace('<vocalsound> ', '')
            sentence=sentence.replace(' <vocalsound>', '')
            sentence=sentence.replace('<vocalsound>', '')
            sentence=sentence.replace('<disfmarker> ', '')
            sentence=sentence.replace(' <disfmarker>', '')
            sentence=sentence.replace('<disfmarker>', '')

            nWord=len(sentence.split())
            nCharacters=len(sentence)
            sentenceList[index]=sentence
            speakerList[index]=mapSpeakerToInteger[speaker]
            nCharacterList[index]=nCharacters
            nWordList[index]=nWord
        

        edge_index=torch.tensor([[]], dtype=torch.long)
        edge_attr=torch.tensor([], dtype=torch.float)

        i=0

        for line in treeStructure:
            linelist=line.split()
            fromNode=int(linelist[0])
            toNode=int(linelist[2])
            addtensor1=torch.tensor([[fromNode, toNode]])
            if i==0:
                edge_index=torch.cat((edge_index,addtensor1), 1)
            else:
                edge_index=torch.cat((edge_index,addtensor1), 0)
            addtensor2=torch.tensor([[toNode, fromNode]])
            edge_index=torch.cat((edge_index,addtensor2), 0)

            matrix[fromNode, toNode]=1
            matrix[toNode, fromNode]=-1

            action=linelist[1]
            action=mapActionToInteger[action]
            edge_attr=torch.cat((edge_attr, torch.tensor([action, action])), 0)

            i+=1

        edge_index=torch.transpose(edge_index, 0, 1)

        
        x=torch.empty((len(transcription), 5), dtype=torch.float)


        for node in transcription:
            index=node["index"]
            
            disf=havedisfmarker[index]
            nWord=nWordList[index]
            nCharacter=nCharacterList[index]
            vocalS=haveVocalSound[index]
            speaker=speakerList[index]

            nodeData=torch.tensor([[disf, nWord, nCharacter, vocalS, speaker]])
            x[index, :]=nodeData


        data=Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_training)

        return data
    
    
    
    
    



