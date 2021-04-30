import math
from utils import constant
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from utils import constant


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(self, num_words, num_chars, word_dim, char_dim, drop_rate, out_dim, word_vectors=None):
        super(Embedding, self).__init__()
        self.word_emb = WordEmbedding(num_words, word_dim, drop_rate, word_vectors=word_vectors)
        self.char_emb = CharacterEmbedding(num_chars, char_dim, drop_rate)
        # output linear layer
        self.linear = Conv1D(in_dim=word_dim + 100, out_dim=out_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, word_ids, char_ids):
        word_emb = self.word_emb(word_ids)  # (batch_size, w_seq_len, word_dim)
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, 100)
        emb = torch.cat([word_emb, char_emb], dim=2)  # (batch_size, w_seq_len, word_dim + 100)
        emb = self.linear(emb)  # (batch_size, w_seq_len, dim)
        return emb


class PositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, inputs, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = inputs.shape[:2]
        if use_cache:
            positions = inputs.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


class VisualAffine(nn.Module):
    def __init__(self, visual_dim, dim, drop_rate=0.0):
        super(VisualAffine, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=1, stride=1, bias=True, padding=0)

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        # assert not torch.isnan(self.linear.conv1d.weight.data).any()
        # assert not torch.isnan(output).any()
        return output

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_rate, num_layers=4):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise_separable_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=dim,
                          padding=kernel_size // 2, bias=False),
                nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
                nn.ReLU(),
            ) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([LayerNorm(normalized_shape=dim, eps=1e-6) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        output = x  # (batch_size, seq_len, dim)
        for idx, conv_layer in enumerate(self.depthwise_separable_conv):
            residual = output
            output = self.layer_norms[idx](output)  # (batch_size, seq_len, dim)
            output = output.transpose(1, 2)  # (batch_size, dim, seq_len)
            output = conv_layer(output)
            output = self.dropout(output)
            output = output.transpose(1, 2) + residual  # (batch_size, seq_len, dim)
        return output


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, drop_rate):
        super(MultiheadAttentionBlock, self).__init__()
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.key = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.value = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.layer_norm1 = LayerNorm(normalized_shape=dim, eps=1e-6)
        self.layer_norm2 = LayerNorm(normalized_shape=dim, eps=1e-6)
        self.out_layer = Conv1D(in_dim=dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask, mask_value=1e-30)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = output + x
        output = self.layer_norm2(residual)
        output = self.dropout(output)
        output = self.out_layer(output)
        output = self.dropout(output) + residual
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0, pos_freeze=False):
        super(FeatureEncoder, self).__init__()
        if not pos_freeze:
            self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim, padding_idx=None)
        else: # for fixed pos_embedding
            self.pos_embedding = PositionalEncoder(dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiheadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        features = self.attention_block(features, mask=mask)  # (batch_size, seq_len, dim)
        return features


class CQAttention(nn.Module):
    def __init__(self, dim, drop_rate=0.0):
        super(CQAttention, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, c_mask, q_mask):
        score = self.trilinear_attention(context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = nn.Softmax(dim=2)(mask_logits(score, q_mask.unsqueeze(1)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=1)(mask_logits(score, c_mask.unsqueeze(2)))  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t), context)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, c2q, torch.mul(context, c2q), torch.mul(context, q2c)], dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand([-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand([-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res


class WeightedPool(nn.Module):
    def __init__(self, dim):
        super(WeightedPool, self).__init__()
        weight = torch.empty(dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, mask):
        alpha = torch.tensordot(x, self.weight, dims=1)  # shape = (batch_size, seq_length, 1)
        alpha = mask_logits(alpha, mask=mask.unsqueeze(2))
        alphas = nn.Softmax(dim=1)(alpha)
        pooled_x = torch.matmul(x.transpose(1, 2), alphas)  # (batch_size, dim, 1)
        pooled_x = pooled_x.squeeze(2)
        return pooled_x


class CQConcatenate(nn.Module):
    def __init__(self, dim):
        super(CQConcatenate, self).__init__()
        self.weighted_pool = WeightedPool(dim=dim)
        self.conv1d = Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, context, query, q_mask):
        pooled_query = self.weighted_pool(query, q_mask)  # (batch_size, dim)
        _, c_seq_len, _ = context.shape
        pooled_query = pooled_query.unsqueeze(1).repeat(1, c_seq_len, 1)  # (batch_size, c_seq_len, dim)
        output = torch.cat([context, pooled_query], dim=2)  # (batch_size, c_seq_len, 2*dim)
        output = self.conv1d(output)
        return output


class ConditionedPredictor(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0, pos_freeze=False):
        super(ConditionedPredictor, self).__init__()
        self.feature_encoder = FeatureEncoder(dim=dim, num_heads=num_heads, max_pos_len=max_pos_len,
                                              kernel_size=kernel_size, num_layers=num_layers, drop_rate=drop_rate,
                                              pos_freeze=pos_freeze)
        self.start_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True))
        self.end_block = nn.Sequential(
            Conv1D(in_dim=2 * dim, out_dim=dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            Conv1D(in_dim=dim, out_dim=1, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, mask):
        start_features = self.feature_encoder(x, mask)  # (batch_size, seq_len, dim)
        end_features = self.feature_encoder(start_features, mask)
        start_features = self.start_block(torch.cat([start_features, x], dim=2))  # (batch_size, seq_len, 1)
        end_features = self.end_block(torch.cat([end_features, x], dim=2))
        start_logits = mask_logits(start_features.squeeze(2), mask=mask)
        end_logits = mask_logits(end_features.squeeze(2), mask=mask)
        return start_logits, end_logits

    @staticmethod
    def extract_index(start_logits, end_logits):
        start_prob = nn.Softmax(dim=1)(start_logits)
        end_prob = nn.Softmax(dim=1)(end_logits)
        outer = torch.matmul(start_prob.unsqueeze(dim=2), end_prob.unsqueeze(dim=1))
        outer = torch.triu(outer, diagonal=0)
        _, start_index = torch.max(torch.max(outer, dim=2)[0], dim=1)  # (batch_size, )
        _, end_index = torch.max(torch.max(outer, dim=1)[0], dim=1)  # (batch_size, )
        return start_index, end_index

    @staticmethod
    def compute_cross_entropy_loss(start_logits, end_logits, start_labels, end_labels):
        start_loss = nn.CrossEntropyLoss(reduction='mean')(start_logits, start_labels)
        end_loss = nn.CrossEntropyLoss(reduction='mean')(end_logits, end_labels)
        return start_loss + end_loss

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 128):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)
        self.pe = nn.Parameter(pe, requires_grad=False)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        # x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        return self.pe[:,:seq_len].cuda()
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x

# added by sicong
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=100, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:len_]
    head = head[:len_]
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1 # just a filler
            if h == 0 or h == min(head):
                root = nodes[i]
            else:
                if h > head.size()[0]:
                    h = head.size()[0]
                nodes[h-1].add_child(nodes[i])

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret

def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, cfgs, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.cfgs = cfgs
        self.layers = num_layers
        self.use_cuda = True
        self.mem_dim = mem_dim
        self.in_dim = cfgs.dim + cfgs.pos_dim + cfgs.ner_dim

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, cfgs.rnn_hidden, cfgs.rnn_layers, batch_first=True, \
                    dropout=cfgs.rnn_dropout, bidirectional=True)
        self.in_dim = cfgs.rnn_hidden * 2
        self.rnn_drop = nn.Dropout(cfgs.rnn_dropout) # use on last layer output

        self.in_drop = nn.Dropout(cfgs.input_dropout)
        self.gcn_drop = nn.Dropout(cfgs.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.UNK_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.cfgs.rnn_hidden, self.cfgs.rnn_layers)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        query_features, masks, pos, ner, deprel, head, words = inputs # unpack

        new_pos_list = []
        for pos_list in pos:
            if len(pos_list) > 20:
                new_pos_list.append(pos_list[:20])
            else:
                pos_list = pos_list.tolist()
                while len(pos_list) < 20:  
                    pos_list.append(0)
                pos_list = torch.tensor(pos_list)
                new_pos_list.append(pos_list)

        new_ner_list = []
        for ner_list in ner:
            if len(ner_list) > 20:
                new_ner_list.append(ner_list[:20])
            else:
                ner_list = ner_list.tolist()
                while len(ner_list) < 20:
                    ner_list.append(0)
                ner_list = torch.tensor(ner_list)
                new_ner_list.append(ner_list)
        pos = torch.stack(new_pos_list)
        ner = torch.stack(new_ner_list)
        pos = pos.cuda("cuda:"+self.cfgs.gpu_idx)
        ner = ner.cuda("cuda:"+self.cfgs.gpu_idx)

        
        # words = torch.tensor(new_word_list)
        # word_embs = self.emb(words)
        embs = [query_features]
        batch = query_features.size()[0]
        if self.cfgs.pos_dim > 0:
            embs += [self.pos_emb(pos)[:batch]]
        if self.cfgs.ner_dim > 0:
            embs += [self.ner_emb(ner)[:batch]]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, len(words)))
        
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            bz = gcn_inputs.size()[0]
            d = torch.zeros([bz,1,gcn_inputs.size()[2]]).cuda("cuda:"+self.cfgs.gpu_idx)
            d2 = torch.zeros([bz,1,1]).cuda("cuda:"+self.cfgs.gpu_idx)
            d2 = d2.type(torch.bool)
            while gcn_inputs.size()[1] < 20:
                gcn_inputs = torch.cat((gcn_inputs, d), dim=1)
                mask = torch.cat((mask, d2), dim=1)
        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class DPFeatureEncoder(nn.Module):
    def __init__(self, dim, num_heads, max_pos_len, kernel_size=7, num_layers=4, drop_rate=0.0, pos_freeze=False):
        super(FeatureEncoder, self).__init__()
        if not pos_freeze:
            self.pos_embedding = PositionalEmbedding(num_embeddings=max_pos_len, embedding_dim=dim, padding_idx=None)
        else: # for fixed pos_embedding
            self.pos_embedding = PositionalEncoder(dim)
        self.conv_block = DepthwiseSeparableConvBlock(dim=dim, kernel_size=kernel_size, drop_rate=drop_rate,
                                                      num_layers=num_layers)
        self.attention_block = MultiheadAttentionBlock(dim=dim, num_heads=num_heads, drop_rate=drop_rate)

    def forward(self, x, mask=None):
        features = x + self.pos_embedding(x)  # (batch_size, seq_len, dim)
        features = self.conv_block(features)  # (batch_size, seq_len, dim)
        features = self.attention_block(features, mask=mask)  # (batch_size, seq_len, dim)
        return features


class DPSentenceEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.dropout = 0.3
        self.max_num_words = 20

        self.bilstm = nn.LSTM(input_size=300,
                              hidden_size=cfgs.dim // 2,
                              num_layers=1,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = F.dropout(x, self.dropout, self.training)
        return x
        