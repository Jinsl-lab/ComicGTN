import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def Glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)
        

def Zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

        
def Segment_Function(x):
    if x <= 500:
        return 0.2, 5
    elif x <= 5000:
        return 0.5, 10
    else:
        return 0.8, 15


class MultiLevelRareLoss(nn.Module):
    def __init__(self, rare_labels, rare_weight, smoothing = 0.1):
        super(MultiLevelRareLoss, self).__init__()
        
        # Registering as a buffer enables the tensor to be correctly transferred to the device.
        self.register_buffer("rare_labels_I", torch.tensor(rare_labels[0]))
        self.register_buffer("rare_labels_II", torch.tensor(rare_labels[1]))
        
        self.rare_weights = {"I": rare_weight[0], "II": rare_weight[1]}
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, node_rep, target):
        device = node_rep.device
        logprobs = F.log_softmax(node_rep, dim = -1)
        weights = torch.ones_like(target, dtype = torch.float32, device = device)
        
        mask_I = torch.isin(target, self.rare_labels_I)
        weights[mask_I] = self.rare_weights["I"]
        mask_II = torch.isin(target, self.rare_labels_II)
        weights[mask_II] = self.rare_weights["II"]
        
        nll_loss = -logprobs.gather(dim = -1, index = target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim = -1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss
    
    
class ImprovedSupConLoss(nn.Module):
    def __init__(self, temperature = 0.1, hard_neg_k = 3, rand_neg_ratio = 0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_neg_k = hard_neg_k          # Number of hard negative samples.
        self.rand_neg_ratio = rand_neg_ratio  # Random negative sample ratio (0~1).

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features_norm = F.normalize(features, dim = 1)
        sim_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
    
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        neg_mask = 1 - mask
        hard_neg_sim, _ = (sim_matrix * neg_mask).topk(k = self.hard_neg_k, dim = 1)
        
        if self.rand_neg_ratio > 0:
            rand_mask = (torch.rand_like(neg_mask) < self.rand_neg_ratio).float()
            rand_neg_mask = neg_mask * rand_mask
            rand_neg_sim = sim_matrix * rand_neg_mask
            mixed_neg_sim = torch.cat([hard_neg_sim, rand_neg_sim], dim = 1)
            
        else:
            mixed_neg_sim = hard_neg_sim
        
        exp_pos = torch.exp(sim_matrix) * mask
        exp_neg = torch.exp(mixed_neg_sim)
        log_prob = torch.log(exp_pos.sum(1) / (exp_pos.sum(1) + exp_neg.sum(1) + 1e-8))
        loss = -log_prob.mean()
        
        return loss