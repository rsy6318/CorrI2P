from numpy import positive
import torch
import torch.nn.functional as F
import numpy as np


def desc_loss(img_features,pc_features,mask,pos_margin=0.1,neg_margin=1.4,log_scale=10,num_kpt=512):
    pos_mask=mask
    neg_mask=1-mask
    #dists=torch.sqrt(torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1))
    dists=1-torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)
    pos=dists-1e5*neg_mask
    pos_weight=(pos-pos_margin).detach()
    pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)
    #pos_weight[pos_weight>0]=1.
    #positive_row=torch.sum((pos[:,:num_kpt,:]-pos_margin)*pos_weight[:,:num_kpt,:],dim=-1)/(torch.sum(pos_weight[:,:num_kpt,:],dim=-1)+1e-8)
    #positive_col=torch.sum((pos[:,:,:num_kpt]-pos_margin)*pos_weight[:,:,:num_kpt],dim=-2)/(torch.sum(pos_weight[:,:,:num_kpt],dim=-2)+1e-8)
    lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
    lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

    neg=dists+1e5*pos_mask
    neg_weight=(neg_margin-neg).detach()
    neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
    #neg_weight[neg_weight>0]=1.
    #negative_row=torch.sum((neg[:,:num_kpt,:]-neg_margin)*neg_weight[:,:num_kpt,:],dim=-1)/torch.sum(neg_weight[:,:num_kpt,:],dim=-1)
    #negative_col=torch.sum((neg[:,:,:num_kpt]-neg_margin)*neg_weight[:,:,:num_kpt],dim=-2)/torch.sum(neg_weight[:,:,:num_kpt],dim=-2)
    lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
    lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

    loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
    loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
    loss=loss_col+loss_row
    
    return torch.mean(loss),dists

def desc_loss2(img_features,pc_features,mask,pos_margin=0.1,neg_margin=1.4,log_scale=10,num_kpt=512):
    pos_mask=mask
    neg_mask=1-mask
    #dists=torch.sqrt(torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1))
    dists=1-torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)
    pos=dists-1e5*neg_mask
    pos_weight=(pos-pos_margin).detach()
    pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)
    #pos_weight[pos_weight>0]=1.
    #positive_row=torch.sum((pos[:,:num_kpt,:]-pos_margin)*pos_weight[:,:num_kpt,:],dim=-1)/(torch.sum(pos_weight[:,:num_kpt,:],dim=-1)+1e-8)
    #positive_col=torch.sum((pos[:,:,:num_kpt]-pos_margin)*pos_weight[:,:,:num_kpt],dim=-2)/(torch.sum(pos_weight[:,:,:num_kpt],dim=-2)+1e-8)
    lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)
    #lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)

    neg=dists+1e5*pos_mask
    neg_weight=(neg_margin-neg).detach()
    neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
    #neg_weight[neg_weight>0]=1.
    #negative_row=torch.sum((neg[:,:num_kpt,:]-neg_margin)*neg_weight[:,:num_kpt,:],dim=-1)/torch.sum(neg_weight[:,:num_kpt,:],dim=-1)
    #negative_col=torch.sum((neg[:,:,:num_kpt]-neg_margin)*neg_weight[:,:,:num_kpt],dim=-2)/torch.sum(neg_weight[:,:,:num_kpt],dim=-2)
    lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
    #lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

    loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
    #loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
    #loss=loss_col+loss_row
    loss=loss_col
    return torch.mean(loss),dists




def det_loss(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
    #score (B,N)
    pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
    diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
    furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
    closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
    loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze()))
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline

def det_loss2(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
    #score (B,N)
    pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
    diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
    furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
    closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
    #loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze())) +torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_inline=torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline


def cal_acc(img_features,pc_features,mask):
    dist=torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1) #(B,N,N)
    furthest_positive,_=torch.max(dist*mask,dim=1)
    closest_negative,_=torch.min(dist+1e5*mask,dim=1)
    '''print(furthest_positive)
    print(closest_negative)
    print(torch.max(torch.sum(mask,dim=1)))
    assert False'''
    diff=furthest_positive-closest_negative
    accuracy=(diff<0).sum(dim=1)/dist.size(1)
    return accuracy