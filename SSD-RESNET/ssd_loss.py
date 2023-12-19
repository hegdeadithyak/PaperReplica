import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_negatives(logits,labels,pos,neg_ratio):

    num_batch,num_anchors,num_classes = logits.shape

    logits = logits.view(-1,num_classes)

    labels = labels.view(-1)
    
    losses = F.cross_entropy(logits,labels,reduction ="none")

    losses = losses.view(num_batch , num_anchors)

    losses[pos]=0

    loss_idx = losses.argsort(1,descending = True)
    rank = loss_idx.argsort(1) 

    num_pos = pos.long().sum(1,keepdim = True)
    num_neg= torch.clamp(neg_ratio * num_pos,max=pos.shape[1]-1)

    neg = rank < num_neg.expand_as(rank)

    return neg


class MultiBoxLoss(nn.Module):

    def __init__(self,num_classes=10,neg_ratio=3):
        super(MultiBoxLoss,self).__init__()
        self.num_classes = num_classes
        self.neg_ratio = neg_ratio

    def forward(self,pred_loc,pred_label,gt_loc,gt_label):

        num_batch = pred_loc.shape[0]

        pos_idx = gt_label>0;
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)


