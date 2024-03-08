from tqdm import tqdm
import torch
import sklearn
import random
import numpy as np

def test(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    f1_score = sklearn.metrics.f1_score(graph.y[mask].cpu(),pred[mask].cpu())

    return acc, f1_score

def train(model, graph, optimizer, criterion, train_mask, val_mask, n_epochs=200, output_evaluation=True):

    train_acc = []
    val_acc = []
    val_F1 = []
    train_F1 = []


    if not output_evaluation:
      iter = tqdm(range(1, n_epochs + 1))
    else:
      iter = range(1, n_epochs + 1)

    for epoch in iter:
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc_val, f1_score_val = test(model, graph, val_mask)
        val_acc.append(acc_val)
        val_F1.append(f1_score_val)

        acc_train, f1_score_train = test(model, graph, train_mask)
        train_acc.append(acc_train)
        train_F1.append(f1_score_train)

        if epoch > 300 and epoch%10==0:
          torch.save(model, f'model_save/epoch_{epoch}.pt')

        if output_evaluation:
          if epoch % 10 == 0:
              print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc_val:.3f}, Val F1-score: {f1_score_val}')

    return model, val_acc, train_acc, val_F1, train_F1

def shuffle_feature(x,column,inside_train=True):
    # Shuffles the features randomly. If it is the BERT features, it shuffles
    # all of the rows together.

    if inside_train:
      X = x.clone()
    else:
      X = x.copy()
    if column == 7:
        idx = np.arange(len(X))
        random.shuffle(idx)
        X[:,column:] = X[idx,column:]
    elif column == 8:
        idx = np.arange(len(X))
        random.shuffle(idx)
        X[:,:4] = X[idx,:4]
    else:
        idx = np.arange(len(X))
        random.shuffle(idx)
        X[:,column] = X[idx,column]
    return X

def train_with_shuffle(model, graph, optimizer, criterion, train_mask, val_mask, n_epochs=200, output_evaluation=True):

    train_acc = []
    val_acc = []
    val_F1 = []
    train_F1 = []

    altered_graph = graph.clone()

    if not output_evaluation:
      iter = tqdm(range(1, n_epochs + 1))
    else:
      iter = range(1, n_epochs + 1)

    for epoch in iter:
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[train_mask], graph.y[train_mask])
        loss.backward()
        optimizer.step()

        acc_val, f1_score_val = test(model, graph, val_mask)
        temp_acc = [acc_val]
        temp_f1 = [f1_score_val]
        for column in range(6):
          altered_graph.x = shuffle_feature(graph.x,column)
          acc_val, f1_score_val = test(model, altered_graph, val_mask)
          temp_acc.append(acc_val)
          temp_f1.append(f1_score_val)
        val_acc.append(temp_acc)
        val_F1.append(temp_f1)

        acc_train, f1_score_train = test(model, graph, train_mask)
        train_acc.append(acc_train)
        train_F1.append(f1_score_train)

        if epoch > 300 and epoch%10==0:
          torch.save(model, f'model_save/epoch_{epoch}.pt')

        if output_evaluation:
          if epoch % 10 == 0:
              print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc_val:.3f}, Val F1-score: {f1_score_val}')

    return model, val_acc, train_acc, val_F1, train_F1