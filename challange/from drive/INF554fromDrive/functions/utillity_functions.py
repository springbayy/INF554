import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
from sklearn import model_selection
import random
import torch

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

def train_and_validation_split(graph_number,val_number,test_number=[]):
    train_number = np.array([i for i in range(1,1+max(graph_number))])
    val_number = np.array(val_number)
    test_number = np.array(test_number)
    train_number = np.delete(train_number,np.where(np.in1d(train_number,val_number)))
    train_number = np.delete(train_number,np.where(np.in1d(train_number,test_number)))

    train_mask = np.in1d(graph_number,train_number)
    val_mask = np.in1d(graph_number,val_number)
    test_mask = np.in1d(graph_number,test_number)
    return train_mask, val_mask, test_mask

def plot_accuracies(val_acc,train_acc):
    plt.plot(val_acc,label='Validation Accuracy')
    plt.plot(train_acc,label='Train Accuracy')
    plt.legend()


def make_submission(json_path: Path = Path("test_labels_naive_baseline.json"),save_name="submission"):
    with open(json_path, "r") as file:
        test_labels = json.load(file)

    file = open(f"results/{save_name}.csv", "w")
    file.write("id,target_feature\n")
    for key, value in test_labels.items():
        u_id = [key + "_" + str(i) for i in range(len(value))]
        target = map(str, value)
        for row in zip(u_id, target):
            file.write(",".join(row))
            file.write("\n")
    file.close()

def split_graph(graph_number,K=5):
    CV = model_selection.KFold(K, shuffle=True)
    indices = np.arange(max(graph_number)+1)
    for _, val_idx in CV.split(indices):
        train_mask, val_mask, _ = train_and_validation_split(graph_number,val_idx)
        return train_mask, val_mask
    
def get_number_of_classifications(graph,model):
    pred = model(graph).argmax(dim=1).cpu().numpy()
    print(f"Number of nodes predicted to be non-important: {(pred==0).sum()}")
    print(f"Number of nodes predicted to be important: {(pred==1).sum()}")