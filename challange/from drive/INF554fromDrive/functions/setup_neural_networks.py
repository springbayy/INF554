from torch_geometric.nn import GCNConv, RGCNConv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

types = np.array(['Continuation', 'Explanation','Elaboration','Acknowledgement',
                      'Comment','Result','Question-answer_pair','Clarification_question',
                      'Contrast','Background','Narration','Alternation','Q-Elab',
                      'Conditional','Correction','Parallel'])
num_relations = len(types)

class RGCN(nn.Module):
    def __init__(self,hidden_units=[256],input_dimension=386,dropout=False):
        super().__init__()
        self.layers = len(hidden_units)
        self.dropout = dropout
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        if  self.layers == 1:
          self.conv1 = RGCNConv(input_dimension, hidden_units[0], num_relations)
          self.conv2 = RGCNConv(hidden_units[0], 2, num_relations)
        elif  self.layers == 2:
          self.conv1 = RGCNConv(input_dimension, hidden_units[0], num_relations)
          self.conv2 = RGCNConv(hidden_units[0], hidden_units[1], num_relations)
          self.conv3 = RGCNConv(hidden_units[1], 2, num_relations)
        else:
          print("Not good")

    def forward(self, data):
        x = data.x
        if self.dropout:
           x = self.dropout1(x)
        edge_index = data.edge_index
        edge_type = data.edge_type

        if self.layers == 1:
          x = F.relu(self.conv1(x, edge_index, edge_type))
          if self.dropout:
            x = self.dropout2(x)
          x = self.conv2(x, edge_index, edge_type)
        elif self.layers == 2:
          x = F.relu(self.conv1(x, edge_index, edge_type))
          if self.dropout:
            x = self.dropout2(x)
          x = F.relu(self.conv2(x, edge_index, edge_type))
          if self.dropout:
            x = self.dropout3(x)
          x = self.conv3(x, edge_index, edge_type)

        #return F.log_softmax(x, dim=1)
        return F.tanh(x)
    

class MLP(nn.Module):
    def __init__(self,input_dimension=386,dropout=False):
        super().__init__()
        if dropout:
          self.layers = nn.Sequential(
          nn.Dropout(0.6),
          nn.Linear(input_dimension, 64),
          nn.ReLU(),
          nn.Dropout(0.6),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Dropout(0.6),
          nn.Linear(32, 2)
          )
        else:
          self.layers = nn.Sequential(
          nn.Linear(input_dimension, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, 2)
          )

    def forward(self, data):
        x = data.x  # only using node features (x)
        output = self.layers(x)
        return output
    

class GCN(nn.Module):
    def __init__(self,hidden_units=[256],input_dimension=386,dropout=False):
        super().__init__()
        self.dropout = dropout
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        self.layers = len(hidden_units)
        if  self.layers == 1:
          self.conv1 = GCNConv(input_dimension, hidden_units[0])
          self.conv2 = GCNConv(hidden_units[0], 2)
        elif  self.layers == 2:
          self.conv1 = GCNConv(input_dimension, hidden_units[0])
          self.conv2 = GCNConv(hidden_units[0], hidden_units[1])
          self.conv3 = GCNConv(hidden_units[1], 2)
        else:
          print("Not good")

    def forward(self, data):
        x = data.x
        if self.dropout:
          x = self.dropout1(x)
        edge_index = data.edge_index

        if self.layers == 1:
          x = F.relu(self.conv1(x, edge_index))
          if self.dropout:
            x = self.dropout2(x)
          x = self.conv2(x, edge_index)
        elif self.layers == 2:
          x = F.relu(self.conv1(x, edge_index))
          if self.dropout:
            x = self.dropout2(x)
          x = F.relu(self.conv2(x, edge_index))
          if self.dropout:
            x = self.dropout3(x)
          x = self.conv3(x, edge_index)

        #return F.log_softmax(x, dim=1)
        return F.tanh(x)
    