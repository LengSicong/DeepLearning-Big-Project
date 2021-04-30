import os
import glob
import pickle as pickle
import numpy as np
from torch.autograd import detect_anomaly
from tqdm import tqdm
import torch
import torch.utils.data
from utils import constant

TACOS = 'tacos'
CHARADES = 'charades'

def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if max_num_clips is None or num_clips <= max_num_clips:
        return visual_feature
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature


def load_video_features(root, max_pos_len):
    video_features = dict()
    filenames = glob.glob(os.path.join(root, "*.npy"))
    data_name = root.split('/')[-1]
    i = 0
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        feature = np.load(filename)
        if max_pos_len is None:
            video_features[video_id] = feature
        else:
            new_feature = visual_feature_sampling(feature, max_num_clips=max_pos_len)
            video_features[video_id] = new_feature
    return video_features


def pad_sequences(sequences, pad_tok=None, max_length=20):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_sequences(sequences, max_length=20, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_sequences(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_sequences(sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length)
    sequence_length, _ = pad_sequences(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_visual_sequence(sequences, max_length=128):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_dim = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_dim], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features

    def __getitem__(self, index):
        record = self.dataset[index]
        # video_feature = self.video_features[record['vid']]
        video_feature = self.video_features["data\\features\\charades\\"+record['vid']]
        start_time, end_time = float(record['s_time']), float(record['e_time'])
        start_index, end_index = int(record['s_ind']), int(record['e_ind'])
        duration = float(record['duration'])
        num_units = int(record['num_units'])
        word_ids, char_ids = record['word_ids'], record['char_ids']
        try:
            ner, pos, dep_head, dep_tree, words = record['ner'], record['pos'], record['dep_head'], record['dep_tree'], record['words']
        except:
            # print("Warning: no GCN input available, setting (ner, pos, ...) to None")
            ner, pos, dep_head, dep_tree, words = None, None, None, None, None
        return video_feature, word_ids, char_ids, start_time, end_time, start_index, end_index, num_units, duration, ner, pos, dep_head, dep_tree, words

    def __len__(self):
        return len(self.dataset)

def map_to_ids(tokens, vocab):
    new_list = []
    for token in tokens:
        ids = [vocab[t] if t in vocab else constant.UNK_ID for t in token]
        new_list.append(ids)
    return tuple(new_list)

def collate_fn(data):
    video_features, word_ids, char_ids, s_times, e_times, s_inds, e_inds, num_units, durations, ner, pos, dep_head, dep_tree, words = zip(*data)
    # process word ids
    word_ids, _ = pad_sequences(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, max_w_seq_len)
    # process char ids
    char_ids, _ = pad_char_sequences(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, max_w_seq_len, max_c_seq_len)
    # process video features
    video_features, _ = pad_visual_sequence(video_features)
    video_features = np.asarray(video_features, dtype=np.float32)  # (batch_size, max_v_seq_len, v_dim)
    # generate times
    s_times = np.asarray(s_times, dtype=np.float32)  # (batch_size, )
    e_times = np.asarray(e_times, dtype=np.float32)
    # generate labels
    s_labels = np.asarray(s_inds, dtype=np.int64)  # (batch_size, )
    e_labels = np.asarray(e_inds, dtype=np.int64)
    # generate num_units (num_units == video_seq_len)
    num_units = np.asarray(num_units, dtype=np.int32)  # (batch_size, )

    try:
        pos = map_to_ids(pos, constant.POS_TO_ID)
        ner = map_to_ids(ner, constant.NER_TO_ID)
        deprel = map_to_ids(dep_tree, constant.DEPREL_TO_ID)
        head = dep_head

        # 16 is the batch size batch_size bz
        pos = get_long_tensor(list(pos), 16)
        ner = get_long_tensor(list(ner), 16)
        head = get_long_tensor(list(head), 16)
        deprel = get_long_tensor(list(deprel), 16)
    except:
        pos, ner, head, deprel = None, None, None, None
            
        
    # convert to tensor
    video_features = torch.tensor(video_features, dtype=torch.float32)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_times = torch.tensor(s_times, dtype=torch.float32)
    e_times = torch.tensor(e_times, dtype=torch.float32)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    num_units = torch.tensor(num_units, dtype=torch.int64)
    durations = torch.tensor(durations, dtype=torch.float32)
    
    # this part is added by xiaoyao for highlignt
    max_length = video_features.size()[1]
    true_batch_size = video_features.size()[0]
    highlight_labels = np.zeros(shape=[true_batch_size, max_length], dtype=np.int32)
    for idx in range(true_batch_size):
        st, et = s_inds[idx], e_inds[idx]
        extend_len = round(0.1*float(et-st+1)) #we fix extend rate as 0.2 for now
        if extend_len>0:
            st_= max(0,st-extend_len)
            et_= min(et+extend_len, max_length-1)
            highlight_labels[idx][st_: et_+ 1] = 1
        else:
            highlight_labels[idx][st: et+ 1] = 1
    highlight_labels = torch.tensor(highlight_labels, dtype=torch.int64)
    
    
    return video_features, word_ids, char_ids, s_times, e_times, s_labels, e_labels, num_units, durations, highlight_labels, pos, ner, deprel, head, words

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_data_loader(cfgs):
    filename = os.path.join(cfgs.save_dir, '_'.join([cfgs.task, str(cfgs.max_pos_len)]) + '.pkl')
    with open(filename, mode='rb') as handle:
        data = pickle.load(handle)
    # load train set
    train_dataset = data['train_set']
    num_train_samples = len(train_dataset)
    # load test set
    test_dataset = data['test_set']
    print("Num test instances:", len(test_dataset))
    num_test_samples = len(test_dataset)
    # load word and character dicts
    num_words = len(data['word_dict'])
    num_chars = len(data['char_dict'])
    # load pre-trained word and video features
    word_vectors = data['word_vector']
    visual_features = load_video_features(os.path.join('data', 'features', cfgs.task), max_pos_len=cfgs.max_pos_len)
    # create dataset
    train_set = Dataset(dataset=train_dataset, video_features=visual_features)
    test_set = Dataset(dataset=test_dataset, video_features=visual_features)
    # create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfgs.batch_size, shuffle=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfgs.batch_size, shuffle=False,
                                              collate_fn=collate_fn)
    return train_loader, test_loader, num_words, num_chars, word_vectors, num_train_samples, num_test_samples


