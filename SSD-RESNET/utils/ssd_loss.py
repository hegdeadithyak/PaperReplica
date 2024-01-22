import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_neg(logits,labels,pos,neg_ratio):
    
    num_batch,num_anchors,num_classes = logits.shape

    logits = logits.view(-1,num_classes)
    labels = labels.view(-1)

    losses = F.cross_entropy(logits,labels,reduction='none')

    losses =losses.view(num_batch,num_anchors)
    
    losses[pos] = 0
    loss_idx= losses.argsort(1,descending=True)
    rank  = loss_idx.argsort(1)

    num_pos = pos.long().sum(1,keepdim = True)
    num_neg = torch.clamp(num_pos*neg_ratio,max =pos.shape[1]-1)
    neg = rank<num_neg.expand_as(rank)

    return neg
