import random
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from torch_geometric.data import Data


def get_speaker(graph_name,train = True):
    speakers = np.array(['PM','ME','UI','ID'])
    speaker_IDs = np.array([1, 2, 3, 4])
    speaker = []

    # Opening JSON file
    if train:
        f = open(f'training/{graph_name}.json')
    else:
        f = open(f'test/{graph_name}.json')

    # returns JSON object as a dictionary
    data = json.load(f)
    for i in range(len(data)):
        name = data[i]['speaker']
        speaker.append([speaker_IDs[speakers==name][0]])
    
    return speaker

def get_edges(graph_name,train=True):
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

    with open(filename) as f:
        for line in f:
            row = line.split()
            n1, n2 = int(row[0]), int(row[2])
            edge_from.append(n1)
            edge_to.append(n2)
            type = types == row[1]
            if not np.any(type):
                print(row[1])
            edge_type.append(arr[type][0])
            
    
    return edge_from, edge_to, edge_type

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
    speaker = get_speaker(graph_name,train=train)

    ## Get edges
    edge_from, edge_to, edge_type = get_edges(graph_name,train=train)
    
    if train:
        ## Get labels
        label = get_labels(graph_name)
        return speaker, edge_from, edge_to, edge_type, label
    
    return speaker, edge_from, edge_to, edge_type 

def setup_graphs(graph_names, train = True, illustrate = False):

    speaker_list = []
    edge_from_list = []
    edge_to_list = []
    edge_type_list = []
    label_list = []
    graph_number = []

    for idx, graph_name in enumerate(graph_names):

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
        graph = Data(x=x, edge_index=edge_index, y=y, edge_type=edge_type)
    else:
        graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    if illustrate:
        g, y = convert_to_networkx(graph)
        plot_graph(g, y)

    return graph, graph_number