import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable
import numpy as np
from model.modules import Embedding, VisualAffine, FeatureEncoder, CQAttention, CQConcatenate, ConditionedPredictor
from transformers import AdamW, get_linear_schedule_with_warmup
from model.contrastive import video_video_loss, video_query_loss
from utils import constant
import os, sys
import copy

INFINITY_NUMBER = 1e12

def max_inter(p, gt_s, gt_e, length):
        individual_loss = []
        for i in range(length.size(0)):
            # vlength = int(length[i])
            index_bs = gt_s[i]
            index_be = gt_e[i]
            ret = torch.log(p[i][index_bs:(index_be+1)])/(max(index_be-index_bs,1))
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.float()
    return inputs * mask + mask_value * (1.0 - mask)

def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False, mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

def mask_softmax(feat, mask):
    return masked_softmax(feat, mask, memory_efficient=False)

def pool(h, mask):
    bz = mask.shape[0]
    seq_len = mask.shape[1]
    dim = h.shape[2]
    mask = mask[:,:h.shape[1]].bool()
    mask = mask.unsqueeze(-1).expand(bz,seq_len,dim)
    if h.shape[0] != mask.shape[0] or h.shape[1] != mask.shape[1]:
        mask = mask[:, : h.shape[1]]
        print("error")
        print('\n')
    h = h.masked_fill(~mask, -INFINITY_NUMBER)
    return torch.max(h, 1)[0]

def build_optimizer_and_scheduler(model, cfgs):
    no_decay = ['bias', 'layer_norm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfgs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfgs.num_train_steps * cfgs.warmup_proportion,
                                                cfgs.num_train_steps)
    return optimizer, scheduler

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    
class LocalizerNetwork(nn.Module):
    def __init__(self, cfgs, word_vectors):
        super(LocalizerNetwork, self).__init__()
        self.cfgs = cfgs
        self.embedding_net = Embedding(num_words=cfgs.num_words, num_chars=cfgs.num_chars, word_dim=cfgs.word_dim,
                                       char_dim=cfgs.char_dim, drop_rate=cfgs.drop_rate, out_dim=cfgs.dim,
                                       word_vectors=word_vectors)
        self.num_stratum = 1       

        if cfgs.dim % self.num_stratum != 0:
            raise Exception("cfgs.num_stratum argument must be a factor of cfgs.dim")
        
        self.video_affine = VisualAffine(visual_dim=cfgs.visual_dim, dim=cfgs.dim, drop_rate=cfgs.drop_rate)
        self.feature_encoder = FeatureEncoder(dim=cfgs.dim, num_heads=cfgs.num_heads, max_pos_len=cfgs.max_pos_len,
                                              kernel_size=7, num_layers=4, drop_rate=cfgs.drop_rate, pos_freeze=cfgs.pos_freeze)
        self.cq_attention = CQAttention(dim=cfgs.dim, drop_rate=cfgs.drop_rate)
        self.cq_concat = CQConcatenate(dim=cfgs.dim)
        self.predictor = ConditionedPredictor(dim=cfgs.dim//self.num_stratum, num_heads=cfgs.num_heads, max_pos_len=cfgs.max_pos_len,
                                              kernel_size=7, num_layers=4, drop_rate=cfgs.drop_rate, pos_freeze=cfgs.pos_freeze)
     
        self.vq_mi = cfgs.vq_mi
        self.vv_mi = cfgs.vv_mi

        self.apply(init_weights)
        
    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask, hightlight_labels, s_labels=None, e_labels=None, pos=None, ner=None, deprel=None, head=None, words=None, is_training = True):
        video_features = self.video_affine(video_features) # linear transformation
        query_features = self.embedding_net(word_ids, char_ids) # linear transformation
        video_features = self.feature_encoder(video_features, mask=v_mask)
        query_features = self.feature_encoder(query_features, mask=q_mask)

        vq_mi_loss = 0
        vv_mi_loss = 0

        if self.vq_mi and is_training:
            video_lens = torch.sum(v_mask,dim=1)
            bz_query_reps = pool(query_features,q_mask)
            for query_reps, v_features, v_len, s, e in zip(bz_query_reps,video_features,video_lens, s_labels, e_labels):
                vq_mi_loss += video_query_loss(v_features,query_reps.view(1,-1),v_len.item(), s, e, self.cfgs.gpu_idx)

        if self.vv_mi and is_training:
            video_lens = torch.sum(v_mask,dim=1)
            for v_features, v_len, s, e in zip(video_features, video_lens, s_labels, e_labels):
                vv_mi_loss += video_video_loss(v_features,v_len.item(),s, e, self.cfgs.gpu_idx)
        
        features = self.cq_attention(video_features, query_features, v_mask, q_mask)
        features = self.cq_concat(features, query_features, q_mask)

        start_logits, end_logits = self.predictor(features, mask=v_mask)
        return start_logits, end_logits,  vq_mi_loss, vv_mi_loss

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
                                                         