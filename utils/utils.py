import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


def compute_metric(pos_score, neg_score):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1))).detach().cpu()
    label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))
    pred_tag = torch.round(torch.sigmoid(pred))
    
    auc = roc_auc_score(label, pred)
    ap = average_precision_score(label, pred)
#     acc = accuracy_score(label, pred_tag)
#     f1 = f1_score(label, pred_tag)

    return auc, ap


def compute_loss(pos_score, neg_score, device):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
    label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))
    
    return F.binary_cross_entropy_with_logits(pred, label.to(device))


def compute_loss_cl(pos_score, neg_score, device):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
    label = torch.cat((torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])))

    return F.binary_cross_entropy_with_logits(pred, label.to(device),reduction="none")

def compute_loss_ce(pos_score, neg_score, device):
    pred = torch.cat((pos_score.squeeze(1), neg_score.squeeze(1)))
    pseudo_y_pos = torch.zeros(pos_score.shape[0])
    pseudo_y_pos[pseudo_y_pos>0.5] = 1.
    pseudo_y_neg = torch.ones(neg_score.shape[0])
    pseudo_y_neg[pseudo_y_neg<0.5] = 0.

    label = torch.cat((pseudo_y_pos, pseudo_y_neg))

    return F.binary_cross_entropy_with_logits(pred, label.to(device))



