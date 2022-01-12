import json
import pickle
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import transformers



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_pkl(load_path):
    with open(load_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data


def write_pkl(pkl_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def compute_sim_mat(a, b):
    a_n, b_n = F.normalize(a, dim=1), F.normalize(b, dim=1)
    sim_mat = a_n @ b_n.T
    return sim_mat


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def auc_at_k(result_list, k):
    """  
    Computing the AUC metric for multi-query evaluation
        
    :param result_list: List, containing evaluation result for some metric 
                        when testing with 1, 2, ... n queries
    """
    return np.trapz(np.array(result_list)[:k]) / (k - 1)


class Tokenizer():
    def __init__(self, model):
        self.model = model
        if self.model == 'CLIP4Clip':
            self.tokenizer = clip.tokenize
        elif self.model == 'FrozenInTime':
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                                "distilbert-base-uncased",
                                TOKENIZERS_PARALLELISM=False)
        else:
            raise ValueError('Model not recognized.') 

    def tokenize(self, captions):
        if self.model == 'CLIP4Clip':
            return self.tokenizer(captions, truncate=True)
        elif self.model  == 'FrozenInTime':
            return self.tokenizer(captions, return_tensors='pt', padding=True, truncation=True)