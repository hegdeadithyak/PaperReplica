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


class MultiBoxLoss(nn.Module):
    def __init__(self,num_classes=3,neg_ratio=3):
        super(MultiBoxLoss,self).__init__()
        self.num_classes = num_classes
        self.neg_ratio  = neg_ratio

    def forward(self,pred_loc,pred_label,gt_loc,gt_label):
        num_batch = pred_loc[0]

        pos_idx = gt_label>0
        pos_loc_idx = pos_idx.unsqueeze(2).expand_as(pred_loc)
        pred_loc_idx = pred_label[pos_loc_idx].view(-1,4)
        gt_loc_idx = gt_label[pos_loc_idx].view(-1,4)

        loc_loss = F.cross_entropy(gt_loc_idx,pred_loc_idx,reduction ="sum")

        logits = pred_loc.detach()
        labels = gt_loc.detach()

        neg_idx=  hard_neg(logits,labels,pos_idx,self.neg_ratio)

        pos_cls_mask = pos_idx.unsqueeze(2).expand_as(pred_label)
        neg_cls_mask = neg_idx.unsqueeze(2).expand_as(pred_label)

        conf_p = pred_label[(pos_cls_mask+neg_cls_mask).gt(0)].view(-1,self.num_classes)
        target = gt_label[(pos_idx+neg_idx).gt(0)]

        cls_loss =F.cross_entropy(conf_p,target,reduction="sum")

        N = pos_idx.long().sum()

        loc_loss/=N
        cls_loss/=N 
        
        return loc_loss,cls_loss
