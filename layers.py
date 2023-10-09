#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import torch
from torch import nn
import math

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(),-loss.sum()

class SoftMaxLoss(nn.Module):
    def __init__(self):
        super(SoftMaxLoss,self).__init__()
        self.classify_loss = nn.NLLLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, output, labels, train=True):
        labels1 = labels.argmax(axis=1)
        out  =  self.log_softmax(output)
        # import pdb;pdb.set_trace()
        cls = self.classify_loss(out,labels1)
        return cls, cls

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        
        # Prevent to overflow
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        cel_sum = 0
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        reshape_out = outs_sig.view(batch_size*num_category,1)
        reshape_lab = labels.view(batch_size*num_category,1)
        
        pos_index = reshape_lab
        neg_index = 1- reshape_lab
        
        pos_loss = torch.mul(pos_index,torch.log(reshape_out + self.epsilon))
        neg_loss = torch.mul(neg_index,torch.log(1- reshape_out + self.epsilon))
        cel_sum = - pos_loss.mean() - neg_loss.mean()
        if torch.isnan(cel_sum):
            import pdb;pdb.set_trace()
        # print(cel_sum,pos_loss.mean(),neg_loss.mean(),alpha_P,alpha_N)
        return cel_sum,cel_sum

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # output = output.view(-1, 5)
        # labels = labels.view(-1, 5)
        # import pdb;pdb.set_trace()
        cls = self.classify_loss(outs,labels)
        # import pdb;pdb.set_trace()
        return cls,cls

class AUCPLoss(nn.Module):
    def __init__(self, num_hard = 0):
        super(AUCPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = 0.1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(2)) / 2
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(2)) / 2
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(2)) / 2
        except:
            import pdb;pdb.set_trace()
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        cls = self.classify_loss(outs,labels) + 0.5 * penalty_term
        # import pdb;pdb.set_trace()
        return cls, 0.5 * penalty_term
    
class AUCHLoss(nn.Module):
    def __init__(self, alpha=0.1,lamb=1, num_hard = 0):
        super(AUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = 0.1
        self.lamb = 1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        cls = self.classify_loss(outs,labels) + self.alpha * penalty_term
        # import pdb;pdb.set_trace()
        return cls, self.alpha * penalty_term
    
class SoftmaxAUCHLoss(nn.Module):
    def __init__(self):
        super(SoftmaxAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = SoftMaxLoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 0.1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        sum_term = 0
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        for k in range(num_category):
            outs = outs_sig[:,k]
            out_pos = outs[labels[:,k] == 1]
            out_neg = outs[labels[:,k] == 0]
            penalty_term_sum = 0

            try:
                num_pos = out_pos.shape[0]
                num_neg = out_neg.shape[0]
                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                    # print("pos")
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                    # print("neg")
                else:
                    trans_pos = out_pos.repeat(num_neg,1)
                    trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
            except:
                import pdb;pdb.set_trace()
                
            sum_term += penalty_term
        # import pdb;pdb.set_trace()
        cls = self.classify_loss(output,labels)[0] + 0.1 * (sum_term / 15)
        # import pdb;pdb.set_trace()
        return cls, 0.1 * penalty_term  
    
class CWAUCHLoss(nn.Module):
    def __init__(self, alpha=1,lamb=1, num_hard = 0):
        super(CWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = CWCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.alpha = alpha
        self.lamb = lamb
        print(alpha,lamb)

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        cls = self.classify_loss(outs,labels)[0] + self.alpha * penalty_term
        # import pdb;pdb.set_trace()
        return cls, self.alpha * penalty_term
    
class FPLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs,pos_loss)
        h_neg_loss = torch.mul(outs,neg_loss)
        
        fpcls = - h_pos_loss.mean() - h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls

class FPLoss1(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs, pos_loss)
        h_neg_loss = torch.mul(outs, neg_loss)
        
        fpcls = - h_pos_loss.mean() - 2 * h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls
    
class RCLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(RCLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs, pos_loss)
        h_neg_loss = torch.mul(outs, neg_loss)
        
        fpcls = - 2 * h_pos_loss.mean() - h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls   

class CWCELoss(nn.Module):
    def __init__(self, num_hard=0):
        super(CWCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        num_neg = neg_labels.sum()
        num_pos = labels.sum()
        
        Beta_P = num_pos / (num_pos + num_neg)
        Beta_N = num_neg / (num_pos + num_neg)
        # import pdb;pdb.set_trace()
        
        pos_loss = torch.mul(labels,torch.log(outs+ self.epsilon))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs+ self.epsilon))
        fpcls = - Beta_N * pos_loss.mean() - Beta_P * neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls , fpcls 

class FocalLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FocalLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        cel_sum = 0
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        reshape_out = outs_sig.view(batch_size*num_category,1)
        reshape_lab = labels.view(batch_size*num_category,1)
        
        pos_index = reshape_lab
        neg_index = 1- reshape_lab
        
        pos_loss = torch.mul(pos_index,torch.log(reshape_out + self.epsilon))
        pos_loss = torch.mul(1-reshape_out,pos_loss)
        neg_loss = torch.mul(neg_index,torch.log(1- reshape_out + self.epsilon))
        neg_loss = torch.mul(reshape_out,neg_loss)
        cel_sum = - pos_loss.mean() - neg_loss.mean()
        if torch.isnan(cel_sum):
            import pdb;pdb.set_trace()
        # print(cel_sum,pos_loss.mean(),neg_loss.mean(),alpha_P,alpha_N)
        return cel_sum , cel_sum 

class CWFocalLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(CWFocalLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        cel_sum = 0
        batch_size = labels.size(0)
        num_category = labels.size(1)
        
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        reshape_out = outs_sig.view(batch_size*num_category,1)
        reshape_lab = labels.view(batch_size*num_category,1)
        
        total_samples = batch_size * num_category
        
        num_P = reshape_lab.sum()
        num_N = total_samples - num_P
        
        alpha_P = num_P / total_samples
        alpha_N = num_N / total_samples
        
        pos_index = reshape_lab
        neg_index = 1- reshape_lab
        
        pos_loss = torch.mul(pos_index,torch.log(reshape_out + self.epsilon))
        pos_loss = torch.mul(1-reshape_out,pos_loss)
        neg_loss = torch.mul(neg_index,torch.log(1- reshape_out + self.epsilon))
        neg_loss = torch.mul(reshape_out,neg_loss)
        cel_sum = - alpha_N * pos_loss.mean() - alpha_P * neg_loss.mean()
        if torch.isnan(cel_sum):
            import pdb;pdb.set_trace()
        # print(cel_sum,pos_loss.mean(),neg_loss.mean(),alpha_P,alpha_N)
        return cel_sum,cel_sum  
    
class MCWCEL(nn.Module):
    def __init__(self):
        super(MCWCEL, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        
        # Prevent to overflow
        self.epsilon = 1e-32
    def forward(self, output, labels, train = True):
        cel_sum = 0
        batch_size = labels.size(0)
        num_category = labels.size(1)
        
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        reshape_out = outs_sig.view(batch_size*num_category,1)
        reshape_lab = labels.view(batch_size*num_category,1)
        
        total_samples = batch_size * num_category
        
        num_P = reshape_lab.sum()
        num_N = total_samples - num_P
        
        alpha_P = num_P / total_samples
        alpha_N = num_N / total_samples
        
        pos_index = reshape_lab
        neg_index = 1- reshape_lab
        
        pos_loss = torch.mul(pos_index,torch.log(reshape_out + self.epsilon))
        neg_loss = torch.mul(neg_index,torch.log(1- reshape_out + self.epsilon))
        cel_sum = - alpha_N * pos_loss.mean() - alpha_P * neg_loss.mean()
        if torch.isnan(cel_sum):
            import pdb;pdb.set_trace()
        # print(cel_sum,pos_loss.mean(),neg_loss.mean(),alpha_P,alpha_N)
        return cel_sum,cel_sum  
    
class FGMCWCEL(nn.Module):
    def __init__(self,alpha=1,lamb=1):
        super(FGMCWCEL, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.lamb = lamb
        print(alpha,lamb)
        
        # Prevent to overflow
        self.epsilon = 1e-32
        
    def forward(self, output, labels, train = True):
        cel_sum = 0
        batch_size = labels.size(0)
        num_category = labels.size(1)
        
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        for i in range(num_category):
            reshape_out = outs_sig[:,i]
            reshape_lab = labels[:,i]

            total_samples = batch_size * num_category

            num_P = reshape_lab.sum()
            num_N = total_samples - num_P

            alpha_P = num_P / total_samples
            alpha_N = num_N / total_samples

            pos_index = reshape_lab
            neg_index = 1- reshape_lab

            pos_loss = torch.mul(pos_index,torch.log(reshape_out + self.epsilon))
            neg_loss = torch.mul(neg_index,torch.log(1- reshape_out + self.epsilon))
            tmp_cel = - alpha_N * pos_loss.mean() - alpha_P * neg_loss.mean()
            if i == 0:
                cel_sum += 4 * tmp_cel
            else:
                cel_sum += tmp_cel
            if torch.isnan(cel_sum):
                import pdb;pdb.set_trace()
            # print(cel_sum,pos_loss.mean(),neg_loss.mean(),alpha_P,alpha_N)
        # import pdb;pdb.set_trace()
        return cel_sum,cel_sum  
    
class FGMCWAUCHLoss(nn.Module):
    def __init__(self, alpha=1,lamb=1):
        super(FGMCWAUCHLoss,  self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = FGMCWCEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = alpha
        self.lamb = lamb
        print(alpha,lamb)

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        sum_term = 0
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        for k in range(num_category):
            outs = outs_sig[:,k]
            out_pos = outs[labels[:,k] == 1]
            out_neg = outs[labels[:,k] == 0]
            penalty_term_sum = 0

            try:
                num_pos = out_pos.shape[0]
                num_neg = out_neg.shape[0]
                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                    # print("pos")
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                    # print("neg")
                else:
                    trans_pos = out_pos.repeat(num_neg,1)
                    trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
            except:
                import pdb;pdb.set_trace()
                
            sum_term += penalty_term
        cls = self.classify_loss(outs_sig,labels)[0] + self.alpha * (sum_term / num_category)
        return cls, self.alpha * penalty_term
    
class MCWAUCHLoss(nn.Module):
    def __init__(self, alpha=1,lamb=1):
        super(MCWAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = MCWCEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 1
        self.lamb = 1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        sum_term = 0
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        for k in range(num_category):
            outs = outs_sig[:,k]
            out_pos = outs[labels[:,k] == 1]
            out_neg = outs[labels[:,k] == 0]
            penalty_term_sum = 0

            try:
                num_pos = out_pos.shape[0]
                num_neg = out_neg.shape[0]
                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                    # print("pos")
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                    # print("neg")
                else:
                    trans_pos = out_pos.repeat(num_neg,1)
                    trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
            except:
                import pdb;pdb.set_trace()
                
            sum_term += penalty_term
        cls = self.classify_loss(outs_sig,labels)[0] + 0.1 * (sum_term / num_category)
        return cls, 0.1 * penalty_term
    
class MAUCHLoss(nn.Module):
    def __init__(self):
        super(MAUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = CEL()
        self.regress_loss = nn.SmoothL1Loss()
        self.alpha = 0.1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        num_category = labels.size(1)
        # import pdb;pdb.set_trace()
        sum_term = 0
        outs_sig = self.sigmoid(output[:,:num_category]).view([batch_size,num_category])
        for k in range(num_category):
            outs = outs_sig[:,k]
            out_pos = outs[labels[:,k] == 1]
            out_neg = outs[labels[:,k] == 0]
            penalty_term_sum = 0

            try:
                num_pos = out_pos.shape[0]
                num_neg = out_neg.shape[0]
                if num_pos == 0:
                    trans_pos = 0
                    trans_neg = out_neg
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                    # print("pos")
                elif num_neg == 0:
                    trans_pos = out_pos
                    trans_neg = 0
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                    # print("neg")
                else:
                    trans_pos = out_pos.repeat(num_neg,1)
                    trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                    penalty_term = torch.mean((1-(trans_pos-trans_neg)))
            except:
                import pdb;pdb.set_trace()
                
            sum_term += penalty_term
        # import pdb;pdb.set_trace()
        cls = self.classify_loss(outs_sig[:,:num_category],labels)[0] + 0.1 * (sum_term / 15)
        # import pdb;pdb.set_trace()
        return cls, 0.1 * penalty_term
    
class FPSimilarityLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs,pos_loss)
        h_neg_loss = torch.mul(outs,neg_loss)
        
        fpcls = - h_pos_loss.mean() - 2 * h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls
    