import networkx as nx
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import to_networkx
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

class Graph:
    def __init__(self,graph_name,train=True) -> None:

        ## Save input
        self.graph_name = graph_name
        self.train = train

        ## Opening JSON file
        if train:
            f = open(f'training/{graph_name}.json')
        else:
            f = open(f'test/{graph_name}.json')
        
        ## Get JSON data 
        self.json_data = json.load(f) # returns JSON object as a dictionary

        ## Get file with graph structure
        if train:
            self.txt_filename = f'training/{graph_name}.txt'
        else:
            self.txt_filename = f'test/{graph_name}.txt'

    def get_features_from_json(self):

        ## Load all utterances and obtain features from them
        bert_embed = []
        self.same_speaker = []
        self.sentence_length = []
        self.disfmarker=[]
        self.gap=[]
        self.vocalsound=[]
        last_speaker = None
        for utterance in self.json_data:
            
            sentence=utterance["text"]

            if '<vocalsound>' in sentence:
                self.vocalsound.append([1])
                self.disfmarker.append0([0])
                self.gap.append([0])
            elif '<disfmarker>' in sentence:
                self.vocalsound.append([0])
                self.disfmarker.append([1])
                self.gap.append([0])
            elif '<gap>' in sentence:
                self.vocalsound.append([0])
                self.disfmarker.append([0])
                self.gap.append([1])


            sentence=sentence.replace('<vocalsound> ', '')
            sentence=sentence.replace(' <vocalsound>', '')
            sentence=sentence.replace('<vocalsound>', '')

            sentence=sentence.replace('<disfmarker> ', '')
            sentence=sentence.replace(' <disfmarker>', '')
            sentence=sentence.replace('<disfmarker>', '')

            sentence=sentence.replace('<gap> ', '')
            sentence=sentence.replace(' <gap>', '')
            sentence=sentence.replace('<gap>', '')

            bert_embed.append(sentence)
            self.sentence_length.append([len(sentence)])
            speaker = utterance['speaker']
            self.same_speaker.append([1 if last_speaker == speaker else 0])

            last_speaker = speaker

        ## Use BERT to get sentence embeddings
        bert = SentenceTransformer('all-mpnet-base-v2')
        self.bert_embed = bert.encode(bert_embed, show_progress_bar=False)

        ## Save number of nodes
        self.number_of_nodes = len(self.sentence_length)

    def get_features_from_txt(self):
        
        ## Setup lists and numpy arrays to store features
        self.edge_from = []
        self.edge_to = []
        self.edge_type = []
        number_of_children = np.zeros(self.number_of_nodes)
        type_of_answer = np.zeros(self.number_of_nodes)

        ## Answer types
        answer_types = np.array(['Elaboration','Continuation','Contrast','Conditional',
                                 'Explanation','Result','Alternation','Background',
                                 'Narration','Clarification_question','Parallel',
                                 'Question-answer_pair','Comment','Q-Elab',
                                 'Correction','Acknowledgement'])
        arr = np.arange(len(answer_types))
        type_values = np.array([0.351194,0.318324,0.313985,0.286149,0.280849,0.259740,
                                0.257225,0.241379,0.235821,0.174955,0.155844,0.120517,
                                0.105898,0.095710,0.048780,0.032745])
        no_connection = 0.069361
        type_of_answer[0] = no_connection
        with open(self.txt_filename) as f:
            for line in f:
                row = line.split()
                n1, n2 = int(row[0]), int(row[2])
                self.edge_from.append(n1)
                number_of_children[n1] += 1
                self.edge_to.append(n2)
                #edge_from.append(n2) # Remove these if we want a directed graph
                #edge_to.append(n1) # Remove these if we want a directed graph
                type = answer_types == row[1]
                self.edge_type.append(arr[type][0])
                #edge_type.append(arr[type][0]) # Remove these if we want a directed graph
                type_of_answer[n2] = type_values[answer_types == row[1]]

        self.number_of_children = number_of_children.reshape(-1,1).tolist()
        self.type_of_answer = type_of_answer.reshape(-1,1).tolist()

    def combine_features(self):
        #self.features = np.hstack((self.sentence_length, self.bert_embed))
        #self.features = np.hstack((self.type_of_answer,self.features))
        #self.features = np.hstack((self.number_of_children,self.features))
        #self.features = np.hstack((self.same_speaker,self.features)).tolist()
        #print(np.array(self.same_speaker).shape,np.array(self.number_of_children).shape,np.array(self.type_of_answer).shape,np.array(self.sentence_length).shape,np.array(self.bert_embed).shape)
        self.features = np.concatenate((self.same_speaker, self.number_of_children, self.type_of_answer, self.sentence_length, self.gap, self.disfmarker, self.vocalsound, self.bert_embed),axis=1).tolist()

    def get_labels(self):

        ## Opening JSON file
        f = open('training_labels.json')

        ## Load JSON file as a directory
        data = json.load(f)

        ## Get the right labels of the graph
        label = data[self.graph_name]

        ## Save labels
        self.labels = [ys for ys in label]

    def get_information(self):
        self.get_features_from_json()
        self.get_features_from_txt()
        self.combine_features()
        if self.train:
            self.get_labels()
            return self.features, self.edge_from, self.edge_to, self.edge_type, self.labels
        else:
            return self.features, self.edge_from, self.edge_to, self.edge_type

def setup_graphs(graph_names, train = True, illustrate = False,device="cpu"):

    feature_list = []
    edge_from_list = []
    edge_to_list = []
    edge_type_list = []
    label_list = []
    graph_number = []

    for idx, graph_name in tqdm(enumerate(graph_names),total=len(graph_names)):

        graph = Graph(graph_name,train=train)

        if train:
            features, edge_from, edge_to, edge_type, labels = graph.get_information()
            label_list += labels
        else:
            features, edge_from, edge_to, edge_type = graph.get_information()

        graph_number += [idx for _ in range(graph.number_of_nodes)]

        edge_from = [e+graph.number_of_nodes for e in edge_from]
        edge_to = [e+graph.number_of_nodes for e in edge_to]
        feature_list += features
        edge_from_list += edge_from
        edge_to_list += edge_to
        edge_type_list += edge_type

    ## Convert to torch tensors
    edge_index = torch.tensor([edge_from_list,edge_to_list], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    x = torch.tensor(feature_list, dtype=torch.float)

    if train:
        ## Get labels
        y = torch.tensor(label_list, dtype=torch.long)
        graph = Data(x=x.to(device), edge_index=edge_index.to(device), y=y.to(device), edge_type=edge_type.to(device))
    else:
        graph = Data(x=x.to(device), edge_index=edge_index.to(device), edge_type=edge_type.to(device))

    return graph, graph_number