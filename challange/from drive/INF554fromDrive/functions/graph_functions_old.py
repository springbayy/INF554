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

def get_features(graph_name,train = True):
    speakers = np.array(['PM','ME','UI','ID'])
    speaker_IDs = np.array([1, 2, 3, 4])
    speaker = []

    # Opening JSON file
    if train:
        f = open(f'training/{graph_name}.json')
    else:
        f = open(f'test/{graph_name}.json')

    last_speaker = None
    same_speaker = []
    # returns JSON object as a dictionary
    data = json.load(f)
    for i in range(len(data)):
        name = data[i]['speaker']
        speaker.append([speaker_IDs[speakers==name][0]])
        same_speaker.append([1 if last_speaker == name else 0])
        last_speaker = name

    #bert = SentenceTransformer('all-MiniLM-L6-v2')
    bert = SentenceTransformer('all-mpnet-base-v2')
    X_training = []
    sentence_length = []
    for idx, utterance in enumerate(data):
        #X_training.append(utterance["speaker"] + ": " + utterance["text"])
        X_training.append(utterance["text"])
        sentence_length.append([len(utterance["text"])])

    #sentence_length = np.array(sentence_length)
    #print(sentence_length)
    #print(speaker)

    X_training = bert.encode(X_training, show_progress_bar=False)

    X_training = np.hstack((sentence_length,X_training))
    #X_training = np.hstack((speaker,X_training))
    X_training = np.hstack((same_speaker,X_training))

    return X_training.tolist()

def get_edges(graph_name,train=True,number_of_nodes=None):
    edge_from = []
    edge_to = []
    edge_type = []

    if train:
        filename = f'training/{graph_name}.txt'
    else:
        filename = f'test/{graph_name}.txt'

    types = np.array(['Continuation', 'Explanation','Elaboration','Acknowledgement',
                      'Comment','Result','Question-answer_pair','Clarification_question',
                      'Contrast','Background','Narration','Alternation','Q-Elab',
                      'Conditional','Correction','Parallel'])
    arr = np.arange(len(types))

    number_of_children = np.zeros(number_of_nodes)
    type_of_answer = np.zeros(number_of_nodes)

    types_ranked = np.array(['Elaboration','Continuation','Contrast','Conditional',
                             'Explanation','Result','Alternation','Background',
                             'Narration','Clarification_question','Parallel',
                             'Question-answer_pair','Comment','Q-Elab',
                             'Correction','Acknowledgement'])
    type_values = np.array([0.351194,0.318324,0.313985,0.286149,0.280849,0.259740,
                            0.257225,0.241379,0.235821,0.174955,0.155844,0.120517,
                            0.105898,0.095710,0.048780,0.032745])
    no_connection = 0.069361
    type_of_answer[0] = no_connection

    with open(filename) as f:
        for line in f:
            row = line.split()
            n1, n2 = int(row[0]), int(row[2])
            edge_from.append(n1)
            number_of_children[n1] += 1
            edge_to.append(n2)
            #edge_from.append(n2) # Remove these if we want a directed graph
            #edge_to.append(n1) # Remove these if we want a directed graph
            type = types == row[1]
            if not np.any(type):
                print(row[1])
            edge_type.append(arr[type][0])
            #edge_type.append(arr[type][0]) # Remove these if we want a directed graph

            type_of_answer[n2] = type_values[types_ranked == row[1]]

    return edge_from, edge_to, edge_type, number_of_children, type_of_answer

def get_labels(graph_name):
    # Opening JSON file
    f = open('training_labels.json')

    # returns JSON object as a dictionary
    data = json.load(f)

    label = data[graph_name]
    label = [ys for ys in label]
    return label

def convert_to_networkx(graph, n_sample=None):

    g = to_networkx(graph, node_attrs=["x"])
    y = graph.y.numpy()

    if n_sample is not None:
        sampled_nodes = random.sample(g.nodes, n_sample)
        g = g.subgraph(sampled_nodes)
        y = y[sampled_nodes]

    return g, y

def plot_graph(g, y):

    plt.figure(figsize=(9, 7))
    nx.draw_planar(g, node_size=5, arrows=True, node_color=y, arrowsize=5,width=0.5)
    plt.show()

def get_graph_features(graph_name, train=True):
    ## Get speaker, 1, 2, 3 and 4 for PM, ME, UI and ID respectively
    features = get_features(graph_name,train=train)

    ## Get edges
    edge_from, edge_to, edge_type, number_of_children, type_of_answer = get_edges(graph_name,train=train,number_of_nodes=len(features))

    ## Add number of children 
    features = np.hstack((number_of_children.reshape(-1,1),np.array(features)))
    features = np.hstack((type_of_answer.reshape(-1,1),features)).tolist()

    if train:
        ## Get labels
        label = get_labels(graph_name)
        return features, edge_from, edge_to, edge_type, label

    return features, edge_from, edge_to, edge_type

def setup_graphs(graph_names, train = True, illustrate = False,device="cpu"):

    speaker_list = []
    edge_from_list = []
    edge_to_list = []
    edge_type_list = []
    label_list = []
    graph_number = []

    for idx, graph_name in tqdm(enumerate(graph_names),total=len(graph_names)):

        if train:
            speaker, edge_from, edge_to, edge_type, label = get_graph_features(graph_name, train=train)
            label_list += label

        else:
            speaker, edge_from, edge_to, edge_type = get_graph_features(graph_name, train=train)

        graph_number += [idx for _ in range(len(speaker))]

        edge_from = [e+len(speaker_list) for e in edge_from]
        edge_to = [e+len(speaker_list) for e in edge_to]
        speaker_list += speaker
        edge_from_list += edge_from
        edge_to_list += edge_to
        edge_type_list += edge_type


    ## Convert to torch tensors
    print(edge_type_list)
    edge_index = torch.tensor([edge_from_list,edge_to_list], dtype=torch.long)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    x = torch.tensor(speaker_list, dtype=torch.float)

    if train:
        ## Get labels
        label = get_labels(graph_name)
        y = torch.tensor(label_list, dtype=torch.long)
        graph = Data(x=x.to(device), edge_index=edge_index.to(device), y=y.to(device), edge_type=edge_type.to(device))
    else:
        graph = Data(x=x.to(device), edge_index=edge_index.to(device), edge_type=edge_type.to(device))

    if illustrate:
        g, y = convert_to_networkx(graph)
        plot_graph(g, y)

    return graph, graph_number